import hashlib
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import Depends, HTTPException, UploadFile
from PIL import Image
from sqlalchemy.exc import SQLAlchemyError

from src.common.database import Database, DatabaseMixin, get_database
from src.common.settings import settings
from src.modules.adps.mysql_object import MySQLObjectCountRepository
from src.modules.detection.models import Detection, DetectionImage, DetectionSession

UPLOAD_DIR = Path(settings.upload_dir)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class DetectionService(DatabaseMixin):
    def __init__(
        self,
        db: Database = Depends(get_database, use_cache=True),
        object_count_repo: MySQLObjectCountRepository = Depends(),
    ) -> None:
        super().__init__(db=db)
        self.object_count_repo = object_count_repo

    def create_detection_session(
        self, image: UploadFile, threshold: float, model_id: Optional[str] = None
    ) -> DetectionSession:
        """
        Create a new detection session and save the uploaded image

        Args:
            image: Uploaded image file
            threshold: Confidence threshold (0.0-1.0)
            model_id: Optional model identifier

        Returns:
            Created DetectionSession instance
        """
        if not (0.0 <= threshold <= 1.0):
            raise HTTPException(
                status_code=400, detail="Threshold must be between 0.0 and 1.0"
            )

        session_id = uuid.uuid4()
        file_path = UPLOAD_DIR / f"{session_id}_{image.filename}"

        # Save image to disk
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Compute image hash for deduplication
        image_hash = self._compute_file_hash(file_path)

        # Get image dimensions
        try:
            img = Image.open(file_path)
            width, height = img.size
        except Exception:
            width, height = 0, 0  # Default if we can't get dimensions

        # Set current timestamp for created_at and updated_at
        now = datetime.now()

        detection_session = DetectionSession(
            id=session_id,
            threshold=threshold,
            image_hash=image_hash,
            image_width=width,
            image_height=height,
            model_id=model_id or "default_model",
            total_objects_detected=0,
            processing_time_ms=0,
            created_at=now,
            updated_at=now,
        )

        try:
            self.db.add(detection_session)
            self.db.flush()  # Flush to get the ID without committing

            detection_image = DetectionImage(
                session_id=session_id,
                storage_type="local",
                image_path=str(file_path),
                original_filename=image.filename,
                mime_type=image.content_type,
                file_size_bytes=file_path.stat().st_size if file_path.exists() else 0,
                storage_metadata={"local_path": str(file_path)},
                updated_at=now,
            )

            self.db.add(detection_image)
            self.db.commit()
            self.db.refresh(detection_session)

            return detection_session

        except SQLAlchemyError as e:
            self.db.rollback()
            # Clean up the uploaded file if there's a database error
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(
                status_code=400, detail=f"Failed to create detection session: {str(e)}"
            )

    def detect_objects(self, session_id: uuid.UUID, predictions: List[Dict]) -> Dict:
        """
        Process detection predictions and store them in the database

        Args:
            session_id: UUID of the detection session
            predictions: List of prediction dictionaries

        Returns:
            Summary of detection results
        """
        object_counts = {}
        total_objects = 0
        start_time = time.time()
        now = datetime.now()

        try:
            for prediction in predictions:
                class_name = prediction["class_name"]
                confidence = prediction["confidence"]
                bbox_x1 = prediction["bbox_x1"]
                bbox_y1 = prediction["bbox_y1"]
                bbox_x2 = prediction["bbox_x2"]
                bbox_y2 = prediction["bbox_y2"]

                if confidence >= self.get_session_threshold(session_id):
                    object_counts[class_name] = object_counts.get(class_name, 0) + 1
                    total_objects += 1

                    detection = Detection(
                        session_id=session_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox_x1=bbox_x1,
                        bbox_y1=bbox_y1,
                        bbox_x2=bbox_x2,
                        bbox_y2=bbox_y2,
                        updated_at=now,
                    )
                    self.db.add(detection)

            # Save counts using the repository
            self.object_count_repo.save_counts(session_id, object_counts)

            # Update session with processing time
            processing_time = int((time.time() - start_time) * 1000)  # Convert to ms
            session = self.get_by_id(DetectionSession, session_id)
            session.total_objects_detected = total_objects
            session.processing_time_ms = processing_time
            session.updated_at = now
            self.db.commit()
            self.db.refresh(session)

            return {
                "session_id": str(session.id),
                "object_counts": object_counts,
                "detections": self.get_detections(session_id),
                "total_objects_detected": total_objects,
                "processing_time_ms": processing_time,
                "threshold_applied": session.threshold,
                "image_dimensions": [session.image_width, session.image_height],
            }

        except SQLAlchemyError as e:
            self.db.rollback()
            raise HTTPException(
                status_code=400, detail=f"Failed to process detections: {str(e)}"
            )

    def get_session_threshold(self, session_id: uuid.UUID) -> float:
        """Get the threshold for a given detection session"""
        session = self.get_by_id(DetectionSession, session_id)
        return session.threshold

    def get_detection_summary(self, session_id: uuid.UUID) -> Dict:
        """Get detection summary for a session"""
        return self.object_count_repo.get_counts(session_id)

    def get_detections(self, session_id: uuid.UUID) -> List[Dict]:
        """Get all detections for a session"""
        detections = self.db.query(Detection).filter_by(session_id=session_id).all()
        return [
            {
                "class_name": det.class_name,
                "confidence": det.confidence,
                "bbox": [det.bbox_x1, det.bbox_y1, det.bbox_x2, det.bbox_y2],
            }
            for det in detections
        ]

    def update_session_dimensions(
        self, session_id: uuid.UUID, width: int, height: int
    ) -> None:
        """Update image dimensions for a detection session"""
        try:
            session = self.get_by_id(DetectionSession, session_id)
            session.image_width = width
            session.image_height = height
            session.updated_at = datetime.now()
            self.db.commit()
            self.db.refresh(session)
        except SQLAlchemyError as e:
            self.db.rollback()
            raise HTTPException(
                status_code=400, detail=f"Failed to update session dimensions: {str(e)}"
            )

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of a file for deduplication"""
        sha256_hash = hashlib.sha256()

        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception:
            # If hash computation fails, use a placeholder
            return f"placeholder-{uuid.uuid4()}"
