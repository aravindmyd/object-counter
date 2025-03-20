from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import constr

from src.common.dependency import DetectorDependency
from src.common.logger import get_api_logger
from src.modules.detection.services import DetectionService

from ..schemas.detect import DetectionResponse

logger = get_api_logger()
router = APIRouter()


@router.post("/", response_model=DetectionResponse)
async def detect_objects(
    image: UploadFile = File(...),
    threshold: float = Form(..., ge=0.0, le=1.0),
    model_id: Optional[constr(min_length=1)] = Form(None),
    detection_service: DetectionService = Depends(),
    detector_dependency: DetectorDependency = Depends(),
):
    """
    Detect objects in an image using the specified model and threshold.

    Args:
        image: The image file to analyze
        threshold: Confidence threshold (0.0-1.0)
        model_id: Optional model identifier

    Returns:
        DetectionResponse with results, counts, and metadata
    """
    logger.info(
        "Processing detection request - image: %s, threshold: %f, model_id: %s",
        image.filename,
        threshold,
        model_id or "default",
    )

    # Get the appropriate detector based on model_id - this is outside the try/except
    # to allow FastAPI to handle validation errors
    detector = detector_dependency.get_detector(model_id)

    try:
        # Create a new detection session
        session = detection_service.create_detection_session(
            image=image, threshold=threshold, model_id=detector.config.model_id
        )

        # Reset file pointer for the detector
        await image.seek(0)

        # Get predictions from model
        predictions = detector.predict(image.file)

        # Convert predictions to the format expected by the API
        results = []
        counts = {}

        for prediction in predictions:
            if prediction.score >= threshold:
                # Add to results
                result = {
                    "class": prediction.class_name,
                    "confidence": prediction.score,
                    "bbox": [
                        prediction.box.xmin,
                        prediction.box.ymin,
                        prediction.box.xmax,
                        prediction.box.ymax,
                    ],
                }
                results.append(result)

                # Update counts
                counts[prediction.class_name] = counts.get(prediction.class_name, 0) + 1

        total_count = sum(counts.values())

        # Save detection results to database
        db_predictions = []
        for prediction in predictions:
            if prediction.score >= threshold:
                db_predictions.append(
                    {
                        "class_name": prediction.class_name,
                        "confidence": prediction.score,
                        "bbox_x1": prediction.box.xmin,
                        "bbox_y1": prediction.box.ymin,
                        "bbox_x2": prediction.box.xmax,
                        "bbox_y2": prediction.box.ymax,
                    }
                )

        detection_service.detect_objects(session.id, db_predictions)

        return DetectionResponse(
            results=results,
            counts=counts,
            total_count=total_count,
            threshold_applied=threshold,
            image_dimensions=[session.image_width, session.image_height],
        )

    except Exception as e:
        logger.info(e)
        logger.error("Error processing detection request: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing detection: {str(e)}",
        )
