import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.app import app
from src.common.database import get_database
from src.common.models import Base
from src.common.settings import settings

# Configuration for test database
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "mysql+pymysql://root:new-password@localhost/test",
)


# Create test settings
@pytest.fixture(scope="session")
def test_settings():
    """Create test settings with test database URL"""
    settings.DATABASE_URL = TEST_DATABASE_URL
    return settings


# Database fixture for integration tests
@pytest.fixture(scope="session")
def test_db_engine(test_settings):
    """Create a test database engine"""
    engine = create_engine(test_settings.DATABASE_URL)
    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def test_db(test_db_engine):
    """Create test database tables and session"""
    # Create all tables
    Base.metadata.create_all(test_db_engine)

    # Create a session
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=test_db_engine
    )
    db = TestingSessionLocal()

    try:
        yield db
    finally:
        db.close()
        # Drop all tables after test
        Base.metadata.drop_all(test_db_engine)


# Override database dependency
@pytest.fixture(scope="function")
def override_db_dependency(test_db):
    """Override the dependency for database"""
    app.dependency_overrides = {}  # Clear any existing overrides

    def get_test_db():
        try:
            yield test_db
        finally:
            pass

    # Override the get_database dependency
    app.dependency_overrides[get_database] = get_test_db

    yield

    # Clear the override after test
    app.dependency_overrides = {}


# TestClient with overridden dependencies
@pytest.fixture(scope="function")
def client(override_db_dependency):
    """Create FastAPI TestClient with overridden dependencies"""
    with TestClient(app) as test_client:
        yield test_client


# Fixture for test image path
@pytest.fixture(scope="session")
def test_image_path():
    """Create path to test images"""
    # You can store test images in a specific directory
    return os.path.join(os.path.dirname(__file__), "test_images")


# Util function to create a test file
def get_test_image_file(filename="test.jpg"):
    """Open a test image file"""
    import io

    from PIL import Image

    # Create a test image
    image = Image.new("RGB", (100, 100), color="red")
    image_io = io.BytesIO()
    image.save(image_io, "JPEG")
    image_io.name = filename
    image_io.seek(0)

    return image_io
