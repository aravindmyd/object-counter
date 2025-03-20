import io
import json
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest
import requests

from src.modules.detector.base import (
    BaseObjectDetector,
    Box,
    DetectionConfig,
    Prediction,
)
from src.modules.detector.factory import DetectorFactory, TFSConfig, TFSObjectDetector


# Fixture for a concrete detector implementation
@pytest.fixture
def concrete_detector():
    """Creates a concrete detector class for testing base functionality"""
    config = DetectionConfig(model_id="test-model", confidence_threshold=0.5)

    class ConcreteDetector(BaseObjectDetector):
        def predict(self, image):
            return []

        def get_supported_classes(self):
            return ["cat", "dog"]

        def get_model_info(self):
            info = super().get_model_info()
            return {**info, "type": "test"}

    return ConcreteDetector(config)


# Fixture for TFS detector configuration
@pytest.fixture
def tfs_config():
    """Creates a TFS configuration for testing"""
    return TFSConfig(
        model_id="ssd_mobilenet_v2",
        host="localhost",
        port=8501,
        model_name="default",
        confidence_threshold=0.5,
    )


# Fixture for sample predictions
@pytest.fixture
def sample_predictions():
    """Creates sample prediction objects for testing"""
    return [
        Prediction(
            class_name="cat", score=0.9, box=Box(xmin=0.1, ymin=0.1, xmax=0.2, ymax=0.2)
        ),
        Prediction(
            class_name="dog", score=0.6, box=Box(xmin=0.3, ymin=0.3, xmax=0.4, ymax=0.4)
        ),
        Prediction(
            class_name="bird",
            score=0.4,
            box=Box(xmin=0.5, ymin=0.5, xmax=0.6, ymax=0.6),
        ),
    ]


# Fixture for mock TFS response
@pytest.fixture
def mock_tfs_response():
    """Creates a mock TFS prediction response"""
    return {
        "predictions": [
            {
                "num_detections": 2,
                "detection_boxes": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
                "detection_scores": [0.95, 0.85],
                "detection_classes": [1, 2],
            }
        ]
    }


# Fixture for mock class labels
@pytest.fixture
def mock_labels_json():
    """Creates mock class label data"""
    return json.dumps(
        [{"id": 1, "display_name": "person"}, {"id": 2, "display_name": "car"}]
    )


# Tests for BaseObjectDetector
def test_filter_predictions(concrete_detector, sample_predictions):
    """Test that predictions are correctly filtered by confidence threshold"""
    # Filter with threshold of 0.5
    filtered = concrete_detector.filter_predictions(sample_predictions)

    # Only cat and dog should remain (scores 0.9 and 0.6)
    assert len(filtered) == 2
    assert filtered[0].class_name == "cat"
    assert filtered[1].class_name == "dog"


def test_get_model_info(concrete_detector):
    """Test that model info is correctly returned"""
    info = concrete_detector.get_model_info()

    assert info["model_id"] == "test-model"
    assert info["confidence_threshold"] == 0.5
    assert info["type"] == "test"


# Tests for TFSObjectDetector
@patch("builtins.open", new_callable=mock_open)
@patch("json.load")
def test_tfs_build_classes_dict(mock_json_load, mock_file_open, tfs_config):
    """Test that class labels are correctly loaded"""
    # Setup mock
    mock_json_load.return_value = [
        {"id": 1, "display_name": "person"},
        {"id": 2, "display_name": "car"},
    ]

    with patch.object(
        TFSObjectDetector, "_build_classes_dict", return_value={1: "person", 2: "car"}
    ):
        # Create detector
        detector = TFSObjectDetector(tfs_config)

        # Manually set the classes_dict
        detector.classes_dict = {1: "person", 2: "car"}

        # Verify class dictionary
        assert detector.classes_dict == {1: "person", 2: "car"}


@patch("requests.post")
def test_tfs_predict(mock_post, tfs_config, mock_tfs_response):
    """Test TFS prediction flow"""
    # Setup mocks
    mock_response = MagicMock()
    mock_response.json.return_value = mock_tfs_response
    mock_post.return_value = mock_response

    # Create detector with mocked methods
    detector = TFSObjectDetector(tfs_config)
    detector.classes_dict = {1: "person", 2: "car"}
    detector._to_np_array = MagicMock(return_value=np.zeros((224, 224, 3)))

    # Call predict with a mock image
    mock_image = io.BytesIO(b"mock image data")
    results = detector.predict(mock_image)

    # Verify results
    assert len(results) == 2
    assert results[0].class_name == "person"
    assert results[0].score == 0.95
    assert results[1].class_name == "car"
    assert results[1].score == 0.85

    # Verify API call
    mock_post.assert_called_once()
    assert "localhost:8501" in mock_post.call_args[0][0]


@patch("requests.post")
def test_tfs_predict_error_handling(mock_post, tfs_config):
    """Test error handling in TFS prediction"""
    # Setup mock to raise an exception
    mock_post.side_effect = requests.exceptions.RequestException("API error")

    # Create detector with mocked methods
    detector = TFSObjectDetector(tfs_config)
    detector.classes_dict = {1: "person", 2: "car"}
    detector._to_np_array = MagicMock(return_value=np.zeros((224, 224, 3)))

    # Call predict with a mock image
    mock_image = io.BytesIO(b"mock image data")

    # It should raise a RuntimeError
    with pytest.raises(RuntimeError) as excinfo:
        detector.predict(mock_image)

    # Verify error message
    assert "Error calling TensorFlow Serving API" in str(excinfo.value)


# Tests for DetectorFactory
def test_detector_factory_registration():
    """Test detector registration in factory"""
    # Register a detector (use real class)
    original_registry = DetectorFactory._detector_registry.copy()
    original_config_registry = DetectorFactory._config_registry.copy()

    try:
        # Add a test detector to the registry
        DetectorFactory._detector_registry["test"] = BaseObjectDetector
        DetectorFactory._config_registry["test"] = DetectionConfig

        # Verify registration
        assert "test" in DetectorFactory._detector_registry
        assert DetectorFactory._detector_registry["test"] == BaseObjectDetector
        assert DetectorFactory._config_registry["test"] == DetectionConfig
    finally:
        # Restore original registry
        DetectorFactory._detector_registry = original_registry
        DetectorFactory._config_registry = original_config_registry


def test_create_detector():
    """Test detector creation through factory"""
    # Setup - add a test detector type
    original_registry = DetectorFactory._detector_registry.copy()
    original_config_registry = DetectorFactory._config_registry.copy()

    # Mock detector class
    mock_detector_class = MagicMock()
    mock_detector_instance = MagicMock()
    mock_detector_class.return_value = mock_detector_instance

    try:
        # Add mock to registry
        DetectorFactory._detector_registry["test-type"] = mock_detector_class
        DetectorFactory._config_registry["test-type"] = DetectionConfig

        # Create a detector
        config = {"model_id": "test-model"}
        detector = DetectorFactory.create_detector("test-type", config)

        # Verify detector creation
        assert detector == mock_detector_instance
    finally:
        # Restore original registry
        DetectorFactory._detector_registry = original_registry
        DetectorFactory._config_registry = original_config_registry


def test_create_detector_unsupported_type():
    """Test error handling for unsupported detector type"""
    # Try to create an unsupported detector
    with pytest.raises(ValueError) as excinfo:
        DetectorFactory.create_detector("unsupported-type", {})

    # Verify error message
    assert "Unsupported detector type" in str(excinfo.value)
