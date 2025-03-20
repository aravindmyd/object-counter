"""Add timestamps to all tables

Revision ID: d0e2d36aa17d
Revises: 9e249477bdba
Create Date: 2025-03-20 10:28:46.312835

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d0e2d36aa17d"
down_revision: Union[str, None] = "9e249477bdba"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLES = ["detection_sessions", "detections", "detection_counts", "detection_images"]


def column_exists(table_name, column_name):
    conn = op.get_bind()
    result = conn.execute(
        sa.text(
            f"SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table_name}' AND COLUMN_NAME = '{column_name}'"
        )
    )
    return result.scalar() > 0


def upgrade():
    for table in TABLES:
        if not column_exists(table, "created_at"):
            op.add_column(
                table,
                sa.Column(
                    "created_at",
                    sa.TIMESTAMP(timezone=False),
                    nullable=False,
                    server_default=sa.text("CURRENT_TIMESTAMP"),
                ),
            )

        if not column_exists(table, "updated_at"):
            op.add_column(
                table,
                sa.Column(
                    "updated_at",
                    sa.TIMESTAMP(timezone=False),
                    nullable=False,
                    server_default=sa.text(
                        "CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"
                    ),
                ),
            )

        if not column_exists(table, "expired_at"):
            op.add_column(
                table,
                sa.Column(
                    "expired_at",
                    sa.TIMESTAMP(timezone=False),
                    nullable=True,
                ),
            )


def downgrade():
    for table in TABLES:
        op.drop_column(table, "created_at")
        op.drop_column(table, "updated_at")
        op.drop_column(table, "expired_at")
