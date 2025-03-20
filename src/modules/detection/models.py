import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import relationship
from sqlalchemy.types import CHAR, TypeDecorator

from src.common.models import Base


# Custom UUID type for MySQL since it doesn't have native UUID support
class MySQLUUID(TypeDecorator):
    impl = CHAR(36)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        elif isinstance(value, uuid.UUID):
            return str(value)
        return value

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if not isinstance(value, uuid.UUID):
            try:
                return uuid.UUID(value)
            except (TypeError, ValueError):
                return None
        return value


class DetectionSession(Base):
    """Model for detection_sessions table"""

    __tablename__ = "detection_sessions"

    # Override id from Base since we're using UUID
    id = Column(MySQLUUID(), primary_key=True, default=uuid.uuid4)
    # Keep base columns but override auto-increment
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        default=datetime.now,
    )
    # Explicitly set updated_at with default value
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        server_onupdate=func.now(),
        default=datetime.now,
    )
    expired_at = Column(DateTime(timezone=True), nullable=True)

    threshold = Column(Float, nullable=False)
    image_hash = Column(String(64), nullable=False, index=True)
    image_width = Column(Integer, nullable=False)
    image_height = Column(Integer, nullable=False)
    model_id = Column(String(50), nullable=False)
    total_objects_detected = Column(Integer, nullable=False)
    processing_time_ms = Column(Integer)

    # Relationships
    detections = relationship(
        "Detection", back_populates="session", cascade="all, delete-orphan"
    )
    detection_counts = relationship(
        "DetectionCount", back_populates="session", cascade="all, delete-orphan"
    )
    image = relationship(
        "DetectionImage",
        back_populates="session",
        uselist=False,
        cascade="all, delete-orphan",
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "threshold >= 0.0 AND threshold <= 1.0", name="threshold_range_check"
        ),
    )


class Detection(Base):
    """Model for detections table"""

    __tablename__ = "detections"

    # Override id from Base since we're using UUID
    id = Column(MySQLUUID(), primary_key=True, default=uuid.uuid4)
    # Explicitly set updated_at with default value
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        server_onupdate=func.now(),
        default=datetime.now,
    )
    expired_at = Column(DateTime(timezone=True), nullable=True)

    session_id = Column(
        MySQLUUID(),
        ForeignKey("detection_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    class_name = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)

    # Relationships
    session = relationship("DetectionSession", back_populates="detections")

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0", name="confidence_range_check"
        ),
        CheckConstraint("bbox_x1 < bbox_x2", name="bbox_x_check"),
        CheckConstraint("bbox_y1 < bbox_y2", name="bbox_y_check"),
    )


class DetectionCount(Base):
    """Model for detection_counts table"""

    __tablename__ = "detection_counts"

    # Override id from Base since we're using UUID
    id = Column(MySQLUUID(), primary_key=True, default=uuid.uuid4)
    # Explicitly set updated_at with default value
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        server_onupdate=func.now(),
        default=datetime.now,
    )
    expired_at = Column(DateTime(timezone=True), nullable=True)

    session_id = Column(
        MySQLUUID(),
        ForeignKey("detection_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    class_name = Column(String(100), nullable=False)
    count = Column(Integer, nullable=False)

    # Relationships
    session = relationship("DetectionSession", back_populates="detection_counts")

    # Constraints
    __table_args__ = (
        CheckConstraint("count > 0", name="count_positive_check"),
        UniqueConstraint("session_id", "class_name", name="unique_class_per_session"),
    )


class DetectionImage(Base):
    """Model for detection_images table"""

    __tablename__ = "detection_images"

    # Use session_id as primary key
    session_id = Column(
        MySQLUUID(),
        ForeignKey("detection_sessions.id", ondelete="CASCADE"),
        primary_key=True,
    )
    # Explicitly set updated_at with default value
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        server_onupdate=func.now(),
        default=datetime.now,
    )
    expired_at = Column(DateTime(timezone=True), nullable=True)

    storage_type = Column(String(20), nullable=False)
    image_path = Column(String(255), nullable=False)
    original_filename = Column(String(255))
    mime_type = Column(String(30), nullable=False)
    file_size_bytes = Column(Integer)
    storage_metadata = Column(JSON)

    # Relationships
    session = relationship("DetectionSession", back_populates="image")
