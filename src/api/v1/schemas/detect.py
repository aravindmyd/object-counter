from pydantic import BaseModel
from typing import List, Dict

class DetectionResponse(BaseModel):
    results: List[Dict[str, float | list]]
    counts: Dict[str, int]
    total_count: int
    threshold_applied: float
    image_dimensions: List[int]