from typing import Dict, List

from pydantic import BaseModel


class DetectionResponse(BaseModel):
    """API response model for object detection"""

    results: List[Dict]
    counts: Dict[str, int]
    total_count: int
    threshold_applied: float
    image_dimensions: List[int]
