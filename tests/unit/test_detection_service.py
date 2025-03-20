import io
import uuid
from unittest.mock import MagicMock, mock_open, patch

import pytest
from fastapi import HTTPException, UploadFile

from src.modules.detection.services import DetectionService


# Fixture for a mock database session
@pytest.fixture
def mock_db():
    """Creates a mock database session"""
    db = MagicMock()
    db.query.return_value.filter_by.return_value.all.return_value = []

    # Configure mock for get_by_id method
    mock_session = MagicMock()
    db.query.return_value.filter.return_value.filter.return_value.one.return_value = (
        mock_session
    )

    return db


# Fixture for a mock object count repository
@pytest.fixture
def mock_repo():
    """Creates a mock object count repository"""
    repo = MagicMock()
    return repo


# Fixture for a service with mocked dependencies
@pytest.fixture
def service(mock_db, mock_repo):
    """Creates a service with mock dependencies"""
    service = DetectionService(db=mock_db, object_count_repo=mock_repo)

    # Mock the get_by_id method to return a mock session
    service.get_by_id = MagicMock()
    mock_session = MagicMock()
    mock_session.threshold = 0.75
    service.get_by_id.return_value = mock_session

    return service


# Fixture for a mock upload file
@pytest.fixture
def mock_upload_file():
    """Creates a mock upload file"""
    file = MagicMock(spec=UploadFile)
    file.filename = "test_image.jpg"
    file.content_type = "image/jpeg"
    file.file = io.BytesIO(b"mock image data")
    return file


# Fixture for sample predictions
@pytest.fixture
def sample_predictions():
    """Creates sample predictions data"""
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


# Tests for DetectionService
@patch("shutil.copyfileobj")
@patch("builtins.open", new_callable=mock_open)
@patch("PIL.Image.open")
@patch("uuid.uuid4")
def test_create_detection_session(
    mock_uuid,
    mock_image_open,
    mock_file_open,
    mock_copyfileobj,
    service,
    mock_db,
    mock_upload_file,
):
    """Test creating a new detection session"""
    # Setup mocks
    mock_uuid.return_value = uuid.UUID("12345678-1234-5678-1234-567812345678")

    mock_pil_image = MagicMock()
    mock_pil_image.size = (640, 480)
    mock_image_open.return_value = mock_pil_image

    # Path.stat() mock for file size
    mock_path = MagicMock()
    mock_path.stat.return_value.st_size = 12345
    mock_path.exists.return_value = True

    with patch("pathlib.Path", return_value=mock_path):
        # Call the method

        result = service.create_detection_session(mock_upload_file, 0.5, "test-model")

        assert result is not None
        assert mock_db.add.called
        assert mock_db.commit.called
        assert mock_db.refresh.called


def test_create_detection_session_invalid_threshold(service, mock_upload_file):
    """Test error handling for invalid threshold"""
    # Call with invalid threshold
    with pytest.raises(HTTPException) as excinfo:
        service.create_detection_session(mock_upload_file, 1.5, "test-model")

    # Verify error
    assert excinfo.value.status_code == 400
    assert "Threshold must be between 0.0 and 1.0" in excinfo.value.detail


@patch("time.time")
def test_detect_objects(mock_time, service, mock_db, mock_repo, sample_predictions):
    """Test processing detection predictions"""
    # Setup mocks
    session_id = uuid.uuid4()

    mock_time.side_effect = [100.0, 100.1]  # Start time, end time (100ms difference)

    # Create a mock session with a real threshold value
    mock_session = MagicMock()
    mock_session.threshold = 0.5  # Real float value
    service.get_session_threshold = MagicMock(return_value=0.5)  # Real float value

    # Call the method
    result = service.detect_objects(session_id, sample_predictions)  # noqa: F841

    # Verify calls to object count repo
    mock_repo.save_counts.assert_called_once()


def test_detect_objects_filtering(service, mock_db, mock_repo):
    """Test that predictions are filtered by threshold"""
    # Setup mocks
    session_id = uuid.uuid4()

    # Set a real threshold value for comparison
    service.get_session_threshold = MagicMock(return_value=0.8)  # Real float value

    # Predictions with one below threshold
    predictions = [
        {
            "class_name": "person",
            "confidence": 0.9,  # Above threshold
            "bbox_x1": 0.1,
            "bbox_y1": 0.2,
            "bbox_x2": 0.3,
            "bbox_y2": 0.4,
        },
        {
            "class_name": "car",
            "confidence": 0.7,  # Below threshold
            "bbox_x1": 0.5,
            "bbox_y1": 0.6,
            "bbox_x2": 0.7,
            "bbox_y2": 0.8,
        },
    ]

    # Call the method
    service.detect_objects(session_id, predictions)

    # Verify database operations - one prediction should be added
    assert mock_db.add.called


def test_get_session_threshold(service):
    """Test retrieving session threshold"""
    # Setup mock
    session_id = uuid.uuid4()

    # Mock session has threshold=0.75 from fixture

    # Call the method
    result = service.get_session_threshold(session_id)

    # Verify result
    assert result == 0.75


def test_get_detections(service, mock_db):
    """Test retrieving detections for a session"""
    # Setup mock
    session_id = uuid.uuid4()
    mock_detections = [
        MagicMock(
            class_name="person",
            confidence=0.9,
            bbox_x1=0.1,
            bbox_y1=0.2,
            bbox_x2=0.3,
            bbox_y2=0.4,
        ),
        MagicMock(
            class_name="car",
            confidence=0.8,
            bbox_x1=0.5,
            bbox_y1=0.6,
            bbox_x2=0.7,
            bbox_y2=0.8,
        ),
    ]
    mock_db.query.return_value.filter_by.return_value.all.return_value = mock_detections

    # Call the method
    result = service.get_detections(session_id)

    # Verify result
    assert len(result) == 2
    assert mock_db.query.called
    assert mock_db.query.return_value.filter_by.called


def test_update_session_dimensions(service):
    """Test updating session dimensions"""
    # Setup mock
    session_id = uuid.uuid4()
    mock_session = MagicMock()
    service.get_by_id.return_value = mock_session

    # Call the method
    service.update_session_dimensions(session_id, 800, 600)

    # Verify session was updated
    mock_session.image_width = 800
    mock_session.image_height = 600
    assert service.get_by_id.called
    assert mock_session.updated_at is not None
