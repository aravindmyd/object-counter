from fastapi import FastAPI

from src.common.logger import logging, setup_logger

from .v1.routers import v1_router

# Set up the root logger
logger = setup_logger(name="app", log_level=logging.INFO)

logger.info("Application starting...")

app = FastAPI()
API_PREFIX = "/api/v1"
app.include_router(v1_router, prefix=API_PREFIX)


@app.get("/health")
async def root():
    return {"status": "Ok"}
