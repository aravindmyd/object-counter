from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel as PydanticBaseModel


class Box(PydanticBaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float


class Prediction(PydanticBaseModel):
    class_name: str
    score: float
    box: Box


# Pydantic models for API responses
class DetectionResult(PydanticBaseModel):
    class_name: str
    confidence: float
    bbox: List[float]


class DetectionSessionResponse(PydanticBaseModel):
    session_id: str
    created_at: datetime
    threshold: float
    model_id: str
    total_objects_detected: int
    processing_time_ms: Optional[int]
    image_dimensions: List[int]


class DetectionResponse(PydanticBaseModel):
    results: List[Dict]
    counts: Dict[str, int]
    total_count: int
    threshold_applied: float
    image_dimensions: List[int]
