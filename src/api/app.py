from fastapi import FastAPI

from src.common.logger import logging, setup_logger

from .v1.routers import v1_router

# Set up the root logger
logger = setup_logger(name="app", log_level=logging.INFO)

logger.info("Application starting...")

# flake8: noqa
app = FastAPI(
    title="Object Counter API",
    description="""
    Object Counter is an API that detects and counts objects in images.

    ## Features
    * Detect objects in images using state-of-the-art models
    * Count specific object types
    * Multiple detection models available
    * Adjustable confidence thresholds
    
    ## Usage
    
    Upload an image to the `/api/v1/detect/` endpoint with optional parameters.
    """,
    version="0.1.0",
)
API_PREFIX = "/api/v1"
app.include_router(v1_router, prefix=API_PREFIX)


@app.get("/health")
async def root():
    return {"status": "Ok"}
