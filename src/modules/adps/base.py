import uuid
from typing import Dict


class ObjectCountRepository:
    """Interface defining operations for an object count repository"""

    def save_counts(self, session_id: uuid.UUID, counts: Dict[str, int]) -> None:
        """Save object counts for a detection session"""

    def get_counts(self, session_id: uuid.UUID) -> Dict[str, int]:
        """Get object counts for a detection session"""

    def get_total_count(self, session_id: uuid.UUID) -> int:
        """Get total object count for a detection session"""

    def get_class_counts_by_date_range(
        self, start_date: str, end_date: str
    ) -> Dict[str, int]:
        """Get total counts by class for a date range"""
