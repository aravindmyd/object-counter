import io
import logging
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import app

logging.basicConfig(level=logging.INFO)

# Skip these tests if E2E_TESTS environment variable is not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("E2E_TESTS"), reason="End-to-end tests disabled"
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def image_path():
    ref_dir = Path(__file__).parent
    return ref_dir.parent.parent / "resources" / "images" / "boy.jpg"


def test_object_detection(client, image_path):
    # Load the image from the resources path
    with open(image_path, "rb") as f:
        image_data = f.read()
    image = io.BytesIO(image_data)

    files = {"file": ("test.jpg", image, "image/jpeg")}
    data = {"threshold": "0.9", "model_name": "resnet"}

    # Make a test request to the object_detection endpoint
    response = client.post("/api/v1/object/object-count", files=files, data=data)

    # Check that the response is valid
    assert response.status_code == 200
    assert response.json() is not None
