import io
import os
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.app import app
from src.modules.detector.base import Box, Prediction


# Create a test client
@pytest.fixture
def client():
    """Create FastAPI TestClient"""
    return TestClient(app)


# Create a test image
@pytest.fixture
def test_image():
    """Create a test image file"""
    # Create a simple test image using PIL
    img = Image.new("RGB", (100, 100), color="red")
    img_io = io.BytesIO()
    img.save(img_io, "JPEG")
    img_io.seek(0)
    return img_io


# Mock the detector and service
@pytest.fixture
def mock_dependencies():
    """Setup mocks for dependencies"""
    # Create paths for the patches - update these paths to match your actual import paths
    patches = [
        patch("src.common.dependency.get_detector"),
        patch("src.api.v1.endpoints.detect.DetectionService"),
    ]

    # Start all patches
    mocks = [p.start() for p in patches]

    # Configure mocks
    mock_detector, mock_service = mocks

    # Configure the detector mock
    detector_instance = MagicMock()
    detector_instance.predict.return_value = [
        Prediction(
            class_name="person",
            score=0.95,
            box=Box(xmin=0.1, ymin=0.2, xmax=0.3, ymax=0.4),
        ),
        Prediction(
            class_name="car",
            score=0.85,
            box=Box(xmin=0.5, ymin=0.6, xmax=0.7, ymax=0.8),
        ),
    ]
    detector_instance.config.model_id = "test-model"
    mock_detector.return_value = detector_instance

    # Configure the service mock
    service_instance = MagicMock()
    mock_session_id = uuid.uuid4()
    mock_session = MagicMock()
    mock_session.id = mock_session_id
    mock_session.image_width = 100  # Match the test image dimensions
    mock_session.image_height = 100  # Match the test image dimensions
    mock_session.threshold = 0.5

    service_instance.create_detection_session.return_value = mock_session
    service_instance.detect_objects.return_value = {
        "results": [
            {"class": "person", "confidence": 0.95, "bbox": [0.1, 0.2, 0.3, 0.4]},
            {"class": "car", "confidence": 0.85, "bbox": [0.5, 0.6, 0.7, 0.8]},
        ],
        "counts": {"person": 1, "car": 1},
        "total_count": 2,
        "threshold_applied": 0.5,
        "image_dimensions": [100, 100],  # Match the test image dimensions
    }
    mock_service.return_value = service_instance

    yield mocks

    # Stop all patches
    for p in patches:
        p.stop()


# Test cases
def test_detect_endpoint_success(client, test_image, mock_dependencies):
    """Test successful object detection request"""
    # Prepare the request with the test image
    files = {"image": ("test_image.jpg", test_image, "image/jpeg")}
    data = {"threshold": 0.5, "model_id": "default"}

    # Send the request
    response = client.post("/api/v1/detect", files=files, data=data)

    # Verify response
    assert response.status_code == 200

    json_response = response.json()
    assert "results" in json_response
    assert "counts" in json_response
    assert "total_count" in json_response
    assert "threshold_applied" in json_response
    assert "image_dimensions" in json_response

    # Verify content
    assert json_response["total_count"] == 2
    assert len(json_response["results"]) == 2
    assert json_response["threshold_applied"] == 0.5
    assert json_response["image_dimensions"] == [
        100,
        100,
    ]  # Match the test image dimensions


def test_detect_endpoint_invalid_threshold(client, test_image, mock_dependencies):
    """Test detection with invalid threshold"""
    # Prepare the request with invalid threshold
    files = {"image": ("test_image.jpg", test_image, "image/jpeg")}
    data = {"threshold": 1.5, "model_id": "default"}  # Invalid: greater than 1.0

    # Configure the first mock (service) to raise exception
    mock_service = mock_dependencies[1]
    service_instance = mock_service.return_value
    service_instance.create_detection_session.side_effect = Exception(
        "Threshold must be between 0.0 and 1.0"
    )

    # Send the request
    response = client.post("/api/v1/detect", files=files, data=data)

    # Verify response is an error
    assert response.status_code >= 400  # Check for any client or server error


def test_detect_endpoint_no_image(client, mock_dependencies):
    """Test detection without providing an image"""
    # Send request without image
    data = {"threshold": 0.5, "model_id": "default"}

    response = client.post("/api/v1/detect", data=data)

    # Verify response is an error
    assert response.status_code >= 400
    assert "detail" in response.json()


def test_models_endpoint(client):
    """Test the endpoint that lists available models"""
    # Use the correct path to your models endpoint based on your specific implementation
    # Let's try with a direct patch of the list_models function
    with patch("src.api.v1.endpoints.models") as mock_list_models:
        mock_list_models.return_value = {
            "models": {
                "default": {
                    "id": "default",
                    "type": "tensorflow-serving",
                    "name": "default",
                    "default_threshold": 0.5,
                }
            },
            "default_model": "default",
        }

        # Send request to the models endpoint
        response = client.get("/api/v1/models")

        # Verify response
        assert response.status_code == 200
        json_response = response.json()
        assert "models" in json_response
        assert "default" in json_response["models"]
        assert json_response["default_model"] == "default"


def test_detect_endpoint_processing_error(client, test_image):
    """Test error handling during image processing using a deliberately bad image"""
    # Create a deliberately invalid image that will cause a processing error
    bad_image = io.BytesIO(b"This is not a valid image file")

    # Prepare the request with the invalid image
    files = {"image": ("bad_image.jpg", bad_image, "image/jpeg")}
    data = {"threshold": 0.5, "model_id": "default"}

    # Send the request - this should fail because the image is not valid
    response = client.post("/api/v1/detect", files=files, data=data)

    # Check that response contains error information
    json_response = response.json()

    # Either check for specific error fields in your API's error response format
    # or simply verify that expected result fields are missing
    assert (
        "error" in json_response
        or "detail" in json_response
        or "results" not in json_response
    )


# Database integration test (requires test database)
@pytest.mark.skipif(
    not os.environ.get("RUN_DB_TESTS"), reason="Database tests disabled"
)
def test_detect_endpoint_database_integration(client, test_image):
    """Test detection with actual database integration"""
    # This test would use a real database connection
    # Only run if RUN_DB_TESTS environment variable is set

    # Prepare the request
    files = {"image": ("test_image.jpg", test_image, "image/jpeg")}
    data = {"threshold": 0.5, "model_id": "default"}

    # Send the request
    response = client.post("/api/v1/detect", files=files, data=data)

    # Verify response
    assert response.status_code == 200
