from typing import Dict, Optional

from fastapi import HTTPException
from fastapi.logger import logger

from src.modules.detector.base import ObjectDetector
from src.modules.detector.factory import get_detector


class DetectorDependency:
    """Dependency to provide configured detector instances"""

    def __init__(self):
        # Load detector configurations
        self.detector_configs = self._load_detector_configs()

    def _load_detector_configs(self) -> Dict[str, Dict]:
        """Load detector configurations from file or environment"""
        # This could be loaded from a config file or environment variables
        return {
            "default": {
                "type": "tensorflow-serving",
                "config": {
                    "host": "localhost",
                    "port": 8501,
                    "model_name": "resnet",
                    "model_id": "ssd_mobilenet_v2",
                    "confidence_threshold": 0.5,
                },
            },
        }

    def get_detector(self, model_id: Optional[str] = None) -> ObjectDetector:
        """
        Get a detector instance based on model_id

        Args:
            model_id: Identifier for the detector configuration to use

        Returns:
            Configured detector instance
        """
        model_id = model_id or "default"

        if model_id not in self.detector_configs:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model: {model_id}. "
                f"Available models: {list(self.detector_configs.keys())}",
            )

        detector_config = self.detector_configs[model_id]

        try:
            return get_detector(
                detector_type=detector_config["type"], config=detector_config["config"]
            )
        except Exception as e:
            logger.error(f"Failed to create detector {model_id}: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to initialize detector: {str(e)}"
            )
