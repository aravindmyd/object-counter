import os
from pathlib import Path

import pytest
import requests
from PIL import Image

from src.common.logger import logging, setup_logger

logger = setup_logger(name="app", log_level=logging.INFO)


# Skip these tests if E2E_TESTS environment variable is not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("E2E_TESTS"), reason="End-to-end tests disabled"
)

# Base URL for your API
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

# Test images directory
TEST_IMAGES_DIR = file_path = (
    Path(__file__).resolve().parent.parent.parent / "resources" / "images"
)


def get_test_image(filename="boy.jpg"):
    """Get a test image file from the test images directory or create one"""
    img_path = os.path.join(TEST_IMAGES_DIR, filename)

    if os.path.exists(img_path):
        return open(img_path, "rb")
    else:
        # Create a test image
        if not os.path.exists(TEST_IMAGES_DIR):
            os.makedirs(TEST_IMAGES_DIR)

        img = Image.new("RGB", (640, 480), color="white")

        # Add some shapes to make the image more interesting
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)
        draw.rectangle([(100, 100), (300, 300)], fill="blue")
        draw.ellipse([(350, 150), (450, 250)], fill="red")

        img.save(img_path)
        return open(img_path, "rb")


def test_list_models():
    """Test listing available models"""
    response = requests.get(f"{API_BASE_URL}/api/v1/models")

    # Check response status
    assert response.status_code == 200

    # Parse response
    data = response.json()

    # Check expected data
    assert "models" in data
    assert "default_model" in data
    assert len(data["models"]) > 0


def test_detect_objects_basic():
    """Test object detection with a basic image"""
    # Get test image
    with get_test_image() as img_file:
        # Prepare request data
        files = {"image": ("test.jpg", img_file, "image/jpeg")}
        data = {
            "threshold": 0.3,  # Lower threshold to catch more objects
        }

        # Send request
        response = requests.post(
            f"{API_BASE_URL}/api/v1/detect", files=files, data=data
        )

        # Check response status
        assert response.status_code == 200

        # Parse response
        data = response.json()

        # Check expected fields
        assert "results" in data
        assert "counts" in data
        assert "total_count" in data
        assert "threshold_applied" in data
        assert "image_dimensions" in data
        assert "person" == data["results"][0]["class"]

        # Log detection results
        logging.info(f"Detected {data['total_count']} objects:")
        for obj_class, count in data["counts"].items():
            logging.info(f"- {obj_class}: {count}")


def test_detect_objects_with_model_id():
    """Test object detection with specific model_id"""
    # First get available models
    models_response = requests.get(f"{API_BASE_URL}/api/v1/models")
    models_data = models_response.json()

    if len(models_data["models"]) < 2:
        pytest.skip("Need at least two models for this test")

    # Get a model ID that's not the default
    default_model = models_data["default_model"]
    model_ids = list(models_data["models"].keys())
    test_model_id = next((m for m in model_ids if m != default_model), default_model)

    # Get test image
    with get_test_image() as img_file:
        # Prepare request data
        files = {"image": ("test.jpg", img_file, "image/jpeg")}
        data = {"threshold": 0.3, "model_id": test_model_id}

        # Send request
        response = requests.post(
            f"{API_BASE_URL}/api/v1/detect", files=files, data=data
        )

        # Check response status
        assert response.status_code == 200

        # Parse response
        data = response.json()

        # Check expected fields
        assert "results" in data
        assert "counts" in data


def test_detect_objects_invalid_threshold():
    """Test object detection with invalid threshold"""
    # Get test image
    with get_test_image() as img_file:
        # Prepare request data with invalid threshold
        files = {"image": ("test.jpg", img_file, "image/jpeg")}
        data = {
            "threshold": 1.5,  # Invalid: greater than 1.0
        }

        # Send request
        response = requests.post(
            f"{API_BASE_URL}/api/v1/detect", files=files, data=data
        )

        # Check response status - should be error
        assert response.status_code == 422

        # Check error message
        error_data = response.json()
        assert "detail" in error_data


def test_detect_objects_invalid_model():
    """Test object detection with invalid model ID"""
    # Get test image
    with get_test_image() as img_file:
        # Prepare request data with invalid model ID
        files = {"image": ("test.jpg", img_file, "image/jpeg")}
        data = {"threshold": 0.5, "model_id": "non_existent_model"}

        # Send request
        response = requests.post(
            f"{API_BASE_URL}/api/v1/detect", files=files, data=data
        )

        # Check response status - should be error
        assert response.status_code == 400

        # Check error message
        error_data = response.json()
        assert "detail" in error_data


def test_detect_objects_real_image():
    """Test object detection with a real image if available"""
    # Check if we have a real test image
    real_image_path = os.path.join(TEST_IMAGES_DIR, "real_test.jpg")

    if not os.path.exists(real_image_path):
        pytest.skip("No real test image available")

    with open(real_image_path, "rb") as img_file:
        # Prepare request data
        files = {"image": ("real_test.jpg", img_file, "image/jpeg")}
        data = {
            "threshold": 0.3,
        }

        # Send request
        response = requests.post(
            f"{API_BASE_URL}/api/v1/detect", files=files, data=data
        )

        # Check response status
        assert response.status_code == 200

        # Parse response
        data = response.json()

        # Check if any objects were detected
        assert "results" in data
        logger.info(f"Detected {data['total_count']} objects in real image:")
        for obj_class, count in data["counts"].items():
            logger.info(f"- {obj_class}: {count}")

        # Real images should have at least some detections
        # This is a rough check - depends on your test image
        if data["total_count"] == 0:
            logger.info("Warning: No objects detected in real image")
