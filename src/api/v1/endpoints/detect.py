import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import confloat, conlist, constr

from src.api.v1.schemas.detect import DetectionResponse
from src.modules.detection.services import DetectionService

router = APIRouter()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@router.get("/detect")
async def detect_objects():
    return {"message": "Detect Objects"}


@router.post("/detect", response_model=DetectionResponse)
async def detect(
    image: UploadFile = File(...),
    threshold: confloat(ge=0.0, le=1.0) = Form(...),
    model_id: Optional[constr(min_length=1)] = Form(None),
    service: DetectionService = Depends(),
):
    logger.info(
        "Received image: %s, threshold: %s, model_id: %s",
        image.filename,
        threshold,
        model_id,
    )
    try:
        response = service.process_image(image, threshold, model_id)
        return response
    except Exception as e:
        logger.error("Error processing image: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal Server Error")
