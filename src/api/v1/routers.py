from fastapi import APIRouter

from .endpoints import detect_router, model_router, object_router

v1_router = APIRouter()
v1_router.include_router(
    detect_router, prefix="/detect", tags=["Object detection (V2) "]
)
v1_router.include_router(object_router, prefix="/object", tags=["Object detection"])
v1_router.include_router(model_router, prefix="/models", tags=["Models"])
