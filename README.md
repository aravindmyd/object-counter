# Object Detection API

This repository provides a robust object detection service that processes images, identifies objects with their confidence scores, and stores the results in a structured format.

## Architecture Overview

The application follows a layered architecture with a plugin-based approach for extensibility:

- **API Layer**: FastAPI endpoints that receive requests and return responses
- **Service Layer**: Core business logic orchestrating the detection process
- **Repository Layer**: Data persistence with MySQL
- **Detector Layer**: Pluggable object detection backends (TensorFlow Serving, etc.)

## Key Features

- **Plugin Architecture**: Support for multiple object detection model backends
- **Relational Database Storage**: Persistence of detection results with MySQL
- **Clean API Design**: Well-structured REST API following best practices
- **Comprehensive Testing**: Unit, integration, and end-to-end tests

## Tech Stack

- **FastAPI**: Web framework for building APIs
- **SQLAlchemy**: ORM for database interactions
- **TensorFlow Serving**: Serving the detection model
- **MySQL**: Relational database for persistence
- **Docker**: Containerization for easy deployment
- **pytest**: Testing framework

## API Endpoints

The API exposes the following endpoints:

### `POST /api/v1/detect`
Process an image and return detected objects

- Accepts multipart form with image file and threshold parameter
- Returns detected objects, counts by class, and metadata

### `GET /api/v1/models`
List available detection models

- Returns information about configured models and their parameters

## Setup and Installation

Setting up the application is straightforward using Docker Compose:

```bash
docker-compose up -d
```

This will:

- Download the object detection model
- Set up MongoDB
- Set up MySQL (for object detection results storage)
- Set up TensorFlow Serving with the detection model
- Run tests to verify functionality
- Expose the API on port 8000

## Usage Examples

### Detecting Objects in an Image

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/api/v1/detect/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@your-file-name.jpg;type=image/jpeg' \
  -F 'threshold=0.5' \
  -F 'model_id=default'
```

### Listing Available Models

```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/api/v1/models' \
  -H 'accept: application/json'
```
# Adding and Switching Models

The application uses a plugin architecture that makes it easy to add new detection models and switch between them.

## Adding a New Model

### Create a new detector implementation

Create a new class that extends the `BaseObjectDetector` interface:

```python
# src/modules/detector/your_model.py
from src.modules.detector.base import BaseObjectDetector, Box, DetectionConfig, Prediction

class YourModelConfig(DetectionConfig):
    # Add model-specific configuration parameters
    custom_param: str
    
class YourModelDetector(BaseObjectDetector):
    def __init__(self, config: YourModelConfig):
        super().__init__(config)
        # Initialize your model
        
    def predict(self, image):
        # Implement prediction logic
        # Return a list of Prediction objects
        
    def get_supported_classes(self):
        # Return list of class names your model can detect
        
    def get_model_info(self):
        # Return model metadata
```

### Register your detector with the factory

Add your detector to the factory in `src/modules/detector/factory.py`:

```python
# Import your detector
from src.modules.detector.your_model import YourModelConfig, YourModelDetector

# Register in the factory
DetectorFactory.register_detector(
    "your-model-type",
    YourModelDetector,
    YourModelConfig
)
```

### Add configuration for your model

Add your model configuration to the detector dependency:

```python
# In src/api/v1/endpoints/detect.py or your config file
detector_configs["your-model-id"] = {
    "type": "your-model-type",
    "config": {
        "model_id": "your-model-id",
        "confidence_threshold": 0.5,
        "custom_param": "custom-value",
        # Add other model-specific parameters
    }
}
```

## Switching Models

When making API requests, you can specify which model to use via the `model_id` parameter:

```bash
# Use the default model
curl -X POST "http://localhost:8000/api/v1/detect" \
  -F "image=@your_image.jpg" \
  -F "threshold=0.5"

# Use a specific model
curl -X POST "http://localhost:8000/api/v1/detect" \
  -F "image=@your_image.jpg" \
  -F "threshold=0.5" \
  -F "model_id=your-model-id"
```

The application will automatically use the appropriate detector implementation based on the `model_id` parameter.



## Development

### Project Structure

```
├── src/
│   ├── api/
│   │   └── v1/
│   │       └── endpoints/       # API endpoints
│   ├── modules/
│   │   ├── detection/
│   │   │   ├── models/          # Database models
│   │   │   └── services/        # Service layer components
│   │   ├── adps/
│   │   │   └── mysql_object.py  # MySQL repository implementation
│   │   └── detector/
│   │       ├── base.py          # Object detector interface
│   │       └── factory.py       # Detector factory pattern
│   └── common/                  # Shared utilities
├── tests/
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── e2e/                     # End-to-end tests
└── docker-compose.yml
```

## Testing

The project includes a comprehensive test suite:

```bash
# Run unit tests
pytest tests/unit

# Run integration tests
pytest tests/integration/

# Run end-to-end tests (requires running API)
E2E_TESTS=1 API_BASE_URL=http://localhost:8000 pytest tests/e2e/

In pyproject.toml, we can enable this variable as well
```

## Model Information

The object detection model used in this example is from IntelAI. The system architecture allows for easy addition of different detection model backends through the plugin pattern.