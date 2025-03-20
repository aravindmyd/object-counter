from typing import Dict, Optional, Type

from .base import BaseObjectDetector, DetectionConfig, ObjectDetector
from .tf_serving import TFSConfig, TFSObjectDetector

# REF FOR FUTURE :
# Add imports for other detector implementations here as they're developed
# from src.modules.detection.detectors.yolo import YOLOConfig, YOLOObjectDetector
# from src.modules.detection.detectors.pytorch import TorchConfig, TorchObjectDetector


class DetectorFactory:
    """
    Factory for creating object detector instances
    """

    # Registry of available detector types
    _detector_registry: Dict[str, Type[BaseObjectDetector]] = {
        "tensorflow-serving": TFSObjectDetector,
        # Add other detector implementations here as they're developed
        # "yolo": YOLOObjectDetector,
        # "pytorch": TorchObjectDetector,
    }

    # Registry of detector configuration types
    _config_registry: Dict[str, Type[DetectionConfig]] = {
        "tensorflow-serving": TFSConfig,
        # Add other configuration types here
        # "yolo": YOLOConfig,
        # "pytorch": TorchConfig,
    }

    @classmethod
    def register_detector(
        cls,
        detector_type: str,
        detector_class: Type[BaseObjectDetector],
        config_class: Type[DetectionConfig],
    ) -> None:
        """
        Register a new detector type

        Args:
            detector_type: String identifier for the detector type
            detector_class: Detector implementation class
            config_class: Configuration class for the detector
        """
        cls._detector_registry[detector_type] = detector_class
        cls._config_registry[detector_type] = config_class

    @classmethod
    def create_detector(cls, detector_type: str, config: Dict) -> ObjectDetector:
        """
        Create a detector instance based on type and config

        Args:
            detector_type: Type of detector to create
            config: Configuration dictionary

        Returns:
            Initialized detector instance

        Raises:
            ValueError: If detector type is not supported
        """
        if detector_type not in cls._detector_registry:
            raise ValueError(f"Unsupported detector type: {detector_type}")

        # Get the appropriate detector and config classes
        detector_class = cls._detector_registry[detector_type]
        config_class = cls._config_registry[detector_type]

        # Create and validate the configuration
        detector_config = config_class(**config)

        # Instantiate the detector with the config
        return detector_class(detector_config)

    @classmethod
    def get_supported_detectors(cls) -> Dict[str, str]:
        """
        Get a list of supported detector types

        Returns:
            Dictionary mapping detector types to descriptions
        """
        return {
            "tensorflow-serving": "TensorFlow Serving object detection models",
            # Add descriptions for other detectors here
        }


# Convenience function for getting a detector instance
def get_detector(
    detector_type: str = "tensorflow-serving", config: Optional[Dict] = None
) -> ObjectDetector:
    """
    Create a detector instance with optional configuration

    Args:
        detector_type: Type of detector to create (default: tensorflow-serving)
        config: Configuration dictionary (optional)

    Returns:
        Initialized detector instance
    """
    # Use default configuration if none provided
    if config is None:
        if detector_type == "tensorflow-serving":
            config = {
                "host": "localhost",
                "port": 8501,
                "model_name": "default",
                "model_id": "ssd_mobilenet_v2",
                "confidence_threshold": 0.5,
            }
        # Add default configs for other detector types

    return DetectorFactory.create_detector(detector_type, config)
