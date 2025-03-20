import uuid
from datetime import datetime
from typing import Dict

from fastapi import Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError

from src.common.database import Database, get_database
from src.modules.detection.models import DetectionCount, DetectionSession

from .base import ObjectCountRepository


class MySQLObjectCountRepository(ObjectCountRepository):
    """
    Implementation of ObjectCountRepository using MySQL
    """

    def __init__(self, db: Database = Depends(get_database, use_cache=True)):
        self.db = db

    def save_counts(self, session_id: uuid.UUID, counts: Dict[str, int]) -> None:
        """
        Save object counts for a detection session using MySQL-specific upsert

        Args:
            session_id: UUID of the detection session
            counts: Dictionary mapping class names to count values
        """
        try:
            # Update total count in the session
            total_count = sum(counts.values())
            session = self.db.query(DetectionSession).filter_by(id=session_id).one()
            session.total_objects_detected = total_count
            session.updated_at = datetime.now()  # Set updated_at explicitly

            # Save each class count using MySQL's INSERT ON DUPLICATE KEY UPDATE syntax
            for class_name, count in counts.items():
                # First check if it exists
                existing = (
                    self.db.query(DetectionCount)
                    .filter_by(session_id=session_id, class_name=class_name)
                    .first()
                )

                if existing:
                    # Update existing count
                    existing.count = existing.count + count
                    existing.updated_at = datetime.now()  # Set updated_at explicitly
                else:
                    # Create new count record with explicit updated_at
                    count_record = DetectionCount(
                        session_id=session_id,
                        class_name=class_name,
                        count=count,
                        updated_at=datetime.now(),  # Set updated_at explicitly
                    )
                    self.db.add(count_record)

            self.db.commit()

        except SQLAlchemyError as e:
            self.db.rollback()
            raise HTTPException(
                status_code=500, detail=f"Failed to save counts: {str(e)}"
            )

    def get_counts(self, session_id: uuid.UUID) -> Dict[str, int]:
        """
        Get object counts for a detection session

        Args:
            session_id: UUID of the detection session

        Returns:
            Dictionary mapping class names to count values
        """
        try:
            counts = (
                self.db.query(DetectionCount).filter_by(session_id=session_id).all()
            )
            return {count.class_name: count.count for count in counts}
        except SQLAlchemyError as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to retrieve counts: {str(e)}"
            )

    def get_total_count(self, session_id: uuid.UUID) -> int:
        """
        Get total object count for a detection session

        Args:
            session_id: UUID of the detection session

        Returns:
            Total count of objects detected
        """
        try:
            session = self.db.query(DetectionSession).filter_by(id=session_id).one()
            return session.total_objects_detected
        except SQLAlchemyError as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to retrieve total count: {str(e)}"
            )

    def get_class_counts_by_date_range(
        self, start_date: str, end_date: str
    ) -> Dict[str, int]:
        """
        Get total counts by class for a date range

        Args:
            start_date: Start date in ISO format (YYYY-MM-DD)
            end_date: End date in ISO format (YYYY-MM-DD)

        Returns:
            Dictionary mapping class names to total counts in the date range
        """
        try:
            counts = (
                self.db.query(
                    DetectionCount.class_name,
                    func.sum(DetectionCount.count).label("total"),
                )
                .join(
                    DetectionSession, DetectionCount.session_id == DetectionSession.id
                )
                .filter(
                    DetectionSession.created_at >= start_date,
                    DetectionSession.created_at <= end_date,
                )
                .group_by(DetectionCount.class_name)
                .all()
            )

            return {count.class_name: count.total for count in counts}
        except SQLAlchemyError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve counts by date range: {str(e)}",
            )
