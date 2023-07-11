from abc import ABC, abstractmethod
from pathlib import Path
from app.models.decision_data_holder import DecisionDataHolder


class AbstractDecisionTable(ABC):
    @staticmethod
    @abstractmethod
    def create_from_csv(filepath: Path) -> "AbstractDecisionTable":
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, ddh: DecisionDataHolder) -> bool:
        raise NotImplementedError()
