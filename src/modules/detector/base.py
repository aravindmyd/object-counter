from abc import ABC, abstractmethod
from typing import BinaryIO, Dict, List

from pydantic import BaseModel


class Box(BaseModel):
    """Bounding box for detected objects"""

    xmin: float
    ymin: float
    xmax: float
    ymax: float


class Prediction(BaseModel):
    """Object detection prediction result"""

    class_name: str
    score: float
    box: Box


class ObjectDetector(ABC):
    """
    Abstract base class for object detection backends.
    All detector implementations must implement these methods.
    """

    @abstractmethod
    def predict(self, image: BinaryIO) -> List[Prediction]:
        """
        Process an image and return object detection predictions

        Args:
            image: Binary image data

        Returns:
            List of Prediction objects
        """

    @abstractmethod
    def get_supported_classes(self) -> List[str]:
        """
        Get the list of object classes supported by this detector

        Returns:
            List of class names
        """

    @abstractmethod
    def get_model_info(self) -> Dict:
        """
        Get information about the model

        Returns:
            Dictionary with model details (name, version, etc.)
        """


class DetectionConfig(BaseModel):
    """Base configuration for detector settings"""

    model_id: str
    confidence_threshold: float = 0.5

    class Config:
        extra = "allow"  # Allow additional fields for specific implementations


class BaseObjectDetector(ObjectDetector):
    """
    Base implementation with common functionality for all detectors
    """

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.model_id = config.model_id
        self.confidence_threshold = config.confidence_threshold

    def filter_predictions(self, predictions: List[Prediction]) -> List[Prediction]:
        """
        Filter predictions by confidence threshold

        Args:
            predictions: List of unfiltered predictions

        Returns:
            Filtered list of predictions
        """
        return [p for p in predictions if p.score >= self.confidence_threshold]

    def get_model_info(self) -> Dict:
        """
        Get basic model information

        Returns:
            Dictionary with model details
        """
        return {
            "model_id": self.model_id,
            "confidence_threshold": self.confidence_threshold,
        }
