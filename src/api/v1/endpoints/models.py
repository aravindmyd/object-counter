from fastapi import APIRouter, Depends, HTTPException
from fastapi.logger import logger

from src.common.dependency import DetectorDependency

router = APIRouter()


@router.get("")
async def list_models(
    detector_dependency: DetectorDependency = Depends(),
):
    """
    List available detection models

    Returns:
        Dictionary of available models and their details
    """
    try:
        # Build a list of available models with basic details
        models = {}

        for model_id, config in detector_dependency.detector_configs.items():
            detector_type = config["type"]
            detector_config = config["config"]

            models[model_id] = {
                "id": model_id,
                "type": detector_type,
                "name": detector_config.get("model_name", model_id),
                "default_threshold": detector_config.get("confidence_threshold", 0.5),
            }

        return {"models": models, "default_model": "default"}

    except Exception as e:
        logger.error("Error listing models: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")
