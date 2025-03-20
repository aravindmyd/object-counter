import io
import uuid
from unittest.mock import MagicMock

import pytest
from fastapi import UploadFile

from src.modules.detector.base import (
    Box,
    Prediction,
)


# Database fixtures
@pytest.fixture
def mock_db():
    """Creates a mock database session"""
    db = MagicMock()
    db.query.return_value.filter_by.return_value.all.return_value = []
    db.query.return_value.filter_by.return_value.one.return_value = None
    db.query.return_value.filter_by.return_value.first.return_value = None
    return db


# Repository fixtures
@pytest.fixture
def mock_repo():
    """Creates a mock object count repository"""
    repo = MagicMock()
    return repo


# Common test data fixtures
@pytest.fixture
def session_id():
    """Creates a test session ID"""
    return uuid.uuid4()


@pytest.fixture
def mock_upload_file():
    """Creates a mock upload file"""
    file = MagicMock(spec=UploadFile)
    file.filename = "test_image.jpg"
    file.content_type = "image/jpeg"
    file.file = io.BytesIO(b"mock image data")
    file.seek = MagicMock()  # Mock the seek method
    return file


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


@pytest.fixture
def sample_db_predictions():
    """Creates sample predictions data for database operations"""
    return [
        {
            "class_name": "person",
            "confidence": 0.95,
            "bbox_x1": 0.1,
            "bbox_y1": 0.2,
            "bbox_x2": 0.3,
            "bbox_y2": 0.4,
        },
        {
            "class_name": "car",
            "confidence": 0.85,
            "bbox_x1": 0.5,
            "bbox_y1": 0.6,
            "bbox_x2": 0.7,
            "bbox_y2": 0.8,
        },
    ]


@pytest.fixture
def sample_counts():
    """Creates sample object counts"""
    return {"person": 3, "car": 2, "dog": 1}
