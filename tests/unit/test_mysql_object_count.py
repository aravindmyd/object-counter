import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy.exc import SQLAlchemyError

from src.modules.adps.mysql_object import MySQLObjectCountRepository
from src.modules.detection.models import DetectionCount, DetectionSession


# Fixture for a mock database session
@pytest.fixture
def mock_db():
    """Creates a mock database session"""
    db = MagicMock()
    db.query.return_value.filter_by.return_value.all.return_value = []
    db.query.return_value.filter_by.return_value.one.return_value = None
    db.query.return_value.filter_by.return_value.first.return_value = None
    return db


# Fixture for a repository with mocked database
@pytest.fixture
def repo(mock_db):
    """Creates a repository with a mock database"""
    repo = MySQLObjectCountRepository(mock_db)
    return repo


# Fixture for a test session ID
@pytest.fixture
def session_id():
    """Creates a test session ID"""
    return uuid.uuid4()


# Fixture for sample object counts
@pytest.fixture
def sample_counts():
    """Creates sample object counts"""
    return {"person": 3, "car": 2, "dog": 1}


# Tests for MySQLObjectCountRepository
def test_save_counts_new_records(repo, mock_db, session_id, sample_counts):
    """Test saving new count records"""
    # Setup mock session
    mock_session = MagicMock()
    mock_session.total_objects_detected = 0
    mock_db.query.return_value.filter_by.return_value.one.return_value = mock_session

    # Setup mock to simulate no existing records
    mock_db.query.return_value.filter_by.return_value.first.return_value = None

    # Call save_counts
    repo.save_counts(session_id, sample_counts)

    # Verify session update
    assert mock_session.total_objects_detected == sum(sample_counts.values())
    assert mock_session.updated_at is not None

    # Verify commit was called
    mock_db.commit.assert_called_once()


def test_save_counts_existing_records(repo, mock_db, session_id, sample_counts):
    """Test updating existing count records"""
    # Setup mock session
    mock_session = MagicMock()
    mock_session.total_objects_detected = 0
    mock_db.query.return_value.filter_by.return_value.one.return_value = mock_session

    # Setup mocks for existing count records
    existing_person = MagicMock()
    existing_person.count = 2

    existing_car = MagicMock()
    existing_car.count = 1

    # Return different existing records based on class_name
    def mock_first_side_effect(**kwargs):
        if kwargs.get("class_name") == "person":
            return existing_person
        elif kwargs.get("class_name") == "car":
            return existing_car
        return None

    mock_db.query.return_value.filter_by.return_value.first.side_effect = (
        mock_first_side_effect
    )

    # Call save_counts
    repo.save_counts(session_id, sample_counts)

    # Verify session update
    assert mock_session.total_objects_detected == sum(sample_counts.values())

    # Verify existing records were updated
    assert existing_person.updated_at is not None
    assert existing_car.updated_at is not None

    # Verify commit was called
    mock_db.commit.assert_called_once()


def test_save_counts_database_error(repo, mock_db, session_id, sample_counts):
    """Test error handling for database errors"""
    # Setup mock to raise an exception
    mock_db.query.side_effect = SQLAlchemyError("Database error")

    # Call save_counts - should raise HTTPException
    with pytest.raises(HTTPException) as excinfo:
        repo.save_counts(session_id, sample_counts)

    # Verify error status and message
    assert excinfo.value.status_code == 500
    assert "Failed to save counts" in excinfo.value.detail

    # Verify rollback was called
    mock_db.rollback.assert_called_once()


def test_get_counts(repo, mock_db, session_id):
    """Test retrieving counts for a session"""
    # Setup mock to return sample counts
    mock_counts = [
        MagicMock(class_name="person", count=3),
        MagicMock(class_name="car", count=2),
    ]
    mock_db.query.return_value.filter_by.return_value.all.return_value = mock_counts

    # Call get_counts
    result = repo.get_counts(session_id)

    # Verify result
    assert result == {"person": 3, "car": 2}

    # Verify query was called with correct session_id
    mock_db.query.assert_called_once_with(DetectionCount)
    mock_db.query.return_value.filter_by.assert_called_once_with(session_id=session_id)


def test_get_counts_database_error(repo, mock_db, session_id):
    """Test error handling for database errors in get_counts"""
    # Setup mock to raise an exception
    mock_db.query.side_effect = SQLAlchemyError("Database error")

    # Call get_counts - should raise HTTPException
    with pytest.raises(HTTPException) as excinfo:
        repo.get_counts(session_id)

    # Verify error status and message
    assert excinfo.value.status_code == 500
    assert "Failed to retrieve counts" in excinfo.value.detail


def test_get_total_count(repo, mock_db, session_id):
    """Test retrieving total count for a session"""
    # Setup mock session
    mock_session = MagicMock()
    mock_session.total_objects_detected = 5
    mock_db.query.return_value.filter_by.return_value.one.return_value = mock_session

    # Call get_total_count
    result = repo.get_total_count(session_id)

    # Verify result
    assert result == 5

    # Verify query was called with correct session_id
    mock_db.query.assert_called_once_with(DetectionSession)
    mock_db.query.return_value.filter_by.assert_called_once_with(id=session_id)


def test_get_total_count_database_error(repo, mock_db, session_id):
    """Test error handling for database errors in get_total_count"""
    # Setup mock to raise an exception
    mock_db.query.side_effect = SQLAlchemyError("Database error")

    # Call get_total_count - should raise HTTPException
    with pytest.raises(HTTPException) as excinfo:
        repo.get_total_count(session_id)

    # Verify error status and message
    assert excinfo.value.status_code == 500
    assert "Failed to retrieve total count" in excinfo.value.detail


@patch("sqlalchemy.func.sum")
def test_get_class_counts_by_date_range(mock_sum, repo, mock_db):
    """Test retrieving counts by date range"""
    # Setup mocks
    mock_sum.return_value = MagicMock(label=MagicMock(return_value="total"))

    mock_counts = [
        MagicMock(class_name="person", total=10),
        MagicMock(class_name="car", total=5),
    ]

    # Configure the chain of method calls on the query
    mock_query = MagicMock()
    mock_db.query.return_value = mock_query
    mock_query.join.return_value = mock_query
    mock_query.filter.return_value = mock_query
    mock_query.group_by.return_value = mock_query
    mock_query.all.return_value = mock_counts

    # Call get_class_counts_by_date_range
    result = repo.get_class_counts_by_date_range("2023-01-01", "2023-01-31")

    # Verify result
    assert result == {"person": 10, "car": 5}

    # Verify query methods were called
    mock_db.query.assert_called_once()
    mock_query.join.assert_called_once()
    mock_query.filter.assert_called_once()
    mock_query.group_by.assert_called_once()
    mock_query.all.assert_called_once()


def test_get_class_counts_by_date_range_database_error(repo, mock_db):
    """Test error handling for database errors in get_class_counts_by_date_range"""
    # Setup mock to raise an exception
    mock_db.query.side_effect = SQLAlchemyError("Database error")

    # Call get_class_counts_by_date_range - should raise HTTPException
    with pytest.raises(HTTPException) as excinfo:
        repo.get_class_counts_by_date_range("2023-01-01", "2023-01-31")

    # Verify error status and message
    assert excinfo.value.status_code == 500
    assert "Failed to retrieve counts by date range" in excinfo.value.detail
