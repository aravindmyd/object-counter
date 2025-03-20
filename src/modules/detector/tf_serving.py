import json
import os
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional

import numpy as np
import requests
from PIL import Image
from pydantic import Field

from .base import BaseObjectDetector, Box, DetectionConfig, Prediction


class TFSConfig(DetectionConfig):
    """Configuration for TensorFlow Serving detector"""

    host: str = Field(default_factory=lambda: os.environ.get("TFS_HOST", "localhost"))
    port: int = Field(default_factory=lambda: int(os.environ.get("TFS_PORT", "8501")))
    model_name: str
    label_map_path: Optional[str] = None

    class Config:
        extra = "allow"


class TFSObjectDetector(BaseObjectDetector):
    """
    Object detector implementation using TensorFlow Serving
    """

    def __init__(self, config: TFSConfig):
        super().__init__(config)
        self.config = config
        # Override host and port from environment variables if available
        self.host = os.environ.get("TFS_HOST", config.host)
        self.port = os.environ.get("TFS_PORT", str(config.port))
        self.model_name = config.model_name
        self.url = f"http://{self.host}:{self.port}/v1/models/{self.model_name}:predict"
        self.classes_dict = self._build_classes_dict(config.label_map_path)

    def predict(self, image: BinaryIO) -> List[Prediction]:
        """
        Process an image and return object detection predictions

        Args:
            image: Binary image data

        Returns:
            List of Prediction objects
        """
        # Convert image to the format expected by TFS
        np_image = self._to_np_array(image)

        # Create the request payload
        predict_request = '{"instances" : %s}' % np.expand_dims(np_image, 0).tolist()

        # Call the TFS API
        try:
            response = requests.post(self.url, data=predict_request)
            response.raise_for_status()  # Raise exception for HTTP errors
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error calling TensorFlow Serving API: {str(e)}")

        # Parse the response
        predictions_json = response.json()

        # Check if we have predictions
        if "predictions" not in predictions_json or not predictions_json["predictions"]:
            return []

        # Convert TFS predictions to domain model
        predictions = self._raw_predictions_to_domain(
            predictions_json["predictions"][0]
        )

        # Filter by confidence threshold
        return self.filter_predictions(predictions)

    def get_supported_classes(self) -> List[str]:
        """Get list of supported object classes"""
        return list(set(self.classes_dict.values()))

    def get_model_info(self) -> Dict:
        """Get TFS model information"""
        base_info = super().get_model_info()
        tfs_info = {
            "backend": "tensorflow-serving",
            "host": self.host,
            "port": self.port,
            "model_name": self.model_name,
            "endpoint": self.url,
            "class_count": len(self.classes_dict),
        }
        return {**base_info, **tfs_info}

    def _build_classes_dict(
        self, label_map_path: Optional[str] = None
    ) -> Dict[int, str]:
        """
        Build a dictionary mapping class IDs to class names

        Args:
            label_map_path: Path to the label map file (optional)

        Returns:
            Dictionary mapping class IDs to names
        """
        # Use provided path or default
        if not label_map_path:
            label_map_path = str(
                Path(__file__).resolve().parent.parent.parent.parent
                / "resources"
                / "model-files"
                / "mscoco_label_map.json"
            )

        try:
            with open(label_map_path) as json_file:
                labels = json.load(json_file)
                return {label["id"]: label["display_name"] for label in labels}
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Error loading class labels: {str(e)}")

    @staticmethod
    def _to_np_array(image: BinaryIO) -> np.ndarray:
        """
        Convert binary image to numpy array for TFS

        Args:
            image: Binary image data

        Returns:
            NumPy array representation
        """
        try:
            image_ = Image.open(image)
            (im_width, im_height) = image_.size
            return (
                np.array(image_.getdata())
                .reshape((im_height, im_width, 3))
                .astype(np.uint8)
            )
        except Exception as e:
            raise RuntimeError(f"Error converting image to numpy array: {str(e)}")

    def _raw_predictions_to_domain(self, raw_predictions: dict) -> List[Prediction]:
        """
        Convert TFS predictions to domain model

        Args:
            raw_predictions: Raw prediction data from TFS

        Returns:
            List of Prediction objects
        """
        predictions = []

        try:
            num_detections = int(raw_predictions.get("num_detections", 0))

            for i in range(num_detections):
                # Extract bounding box
                detection_box = raw_predictions["detection_boxes"][i]
                box = Box(
                    ymin=detection_box[0],
                    xmin=detection_box[1],
                    ymax=detection_box[2],
                    xmax=detection_box[3],
                )

                # Extract score and class
                detection_score = raw_predictions["detection_scores"][i]
                detection_class = int(raw_predictions["detection_classes"][i])

                # Get class name from mapping
                class_name = self.classes_dict.get(
                    detection_class, f"unknown-{detection_class}"
                )

                # Create prediction object
                predictions.append(
                    Prediction(class_name=class_name, score=detection_score, box=box)
                )

            return predictions

        except (KeyError, IndexError, ValueError) as e:
            raise RuntimeError(f"Error parsing TFS predictions: {str(e)}")
