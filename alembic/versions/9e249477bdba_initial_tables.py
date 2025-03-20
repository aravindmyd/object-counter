from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.mysql import JSON

from alembic import op

# revision identifiers, used by Alembic
revision: str = "9e249477bdba"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Create detection_sessions table
    op.create_table(
        "detection_sessions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=False),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("image_hash", sa.String(64), nullable=False),
        sa.Column("image_width", sa.Integer(), nullable=False),
        sa.Column("image_height", sa.Integer(), nullable=False),
        sa.Column("model_id", sa.String(50), nullable=False),
        sa.Column("total_objects_detected", sa.Integer(), nullable=False),
        sa.Column("processing_time_ms", sa.Integer(), nullable=True),
        sa.CheckConstraint("threshold BETWEEN 0.0 AND 1.0", name="threshold_check"),
    )

    # Create detections table
    op.create_table(
        "detections",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("session_id", sa.String(36), nullable=False),
        sa.Column("class_name", sa.String(100), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("bbox_x1", sa.Float(), nullable=False),
        sa.Column("bbox_y1", sa.Float(), nullable=False),
        sa.Column("bbox_x2", sa.Float(), nullable=False),
        sa.Column("bbox_y2", sa.Float(), nullable=False),
        sa.CheckConstraint("confidence BETWEEN 0.0 AND 1.0", name="confidence_check"),
        sa.CheckConstraint("bbox_x1 < bbox_x2", name="x_coord_check"),
        sa.CheckConstraint("bbox_y1 < bbox_y2", name="y_coord_check"),
        sa.ForeignKeyConstraint(
            ["session_id"], ["detection_sessions.id"], ondelete="CASCADE"
        ),
    )

    # Create detection_counts table
    op.create_table(
        "detection_counts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("session_id", sa.String(36), nullable=False),
        sa.Column("class_name", sa.String(100), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False),
        sa.CheckConstraint("count > 0", name="count_check"),
        sa.ForeignKeyConstraint(
            ["session_id"], ["detection_sessions.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint("session_id", "class_name", name="uq_session_class"),
    )

    # Create detection_images table
    op.create_table(
        "detection_images",
        sa.Column("session_id", sa.String(36), primary_key=True),
        sa.Column("storage_type", sa.String(20), nullable=False),
        sa.Column("image_path", sa.String(255), nullable=False),
        sa.Column("original_filename", sa.String(255)),
        sa.Column("mime_type", sa.String(30), nullable=False),
        sa.Column("file_size_bytes", sa.Integer()),
        sa.Column("storage_metadata", JSON, nullable=True),
        sa.ForeignKeyConstraint(
            ["session_id"], ["detection_sessions.id"], ondelete="CASCADE"
        ),
    )

    # Create indexes
    op.create_index("idx_detections_session_id", "detections", ["session_id"])
    op.create_index("idx_detections_class_name", "detections", ["class_name"])
    op.create_index(
        "idx_detection_counts_session_id", "detection_counts", ["session_id"]
    )
    op.create_index(
        "idx_detection_counts_class_name", "detection_counts", ["class_name"]
    )
    op.create_index(
        "idx_detection_sessions_created_at", "detection_sessions", ["created_at"]
    )
    op.create_index(
        "idx_detection_sessions_image_hash", "detection_sessions", ["image_hash"]
    )


def downgrade():
    # Drop all tables in reverse order of creation
    op.drop_index("idx_detection_sessions_image_hash")
    op.drop_index("idx_detection_sessions_created_at")
    op.drop_index("idx_detection_counts_class_name")
    op.drop_index("idx_detection_counts_session_id")
    op.drop_index("idx_detections_class_name")
    op.drop_index("idx_detections_session_id")

    op.drop_table("detection_images")
    op.drop_table("detection_counts")
    op.drop_table("detections")
    op.drop_table("detection_sessions")
