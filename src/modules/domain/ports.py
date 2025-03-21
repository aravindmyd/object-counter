from abc import ABC, abstractmethod
from typing import BinaryIO, List

from src.modules.domain.models import ObjectCount, Prediction


class ObjectDetector(ABC):
    @abstractmethod
    def predict(self, image: BinaryIO) -> List[Prediction]:
        raise NotImplementedError


class ObjectCountRepo(ABC):
    @abstractmethod
    def read_values(self, object_classes: List[str] = None) -> List[ObjectCount]:
        raise NotImplementedError

    @abstractmethod
    def update_values(self, new_values: List[ObjectCount]):
        raise NotImplementedError
