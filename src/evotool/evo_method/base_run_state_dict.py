from abc import ABC, abstractmethod
import json


class BaseRunStateDict(ABC):
    def __init__(
            self, task_info:dict
    ):
        self.task_info = task_info

    @abstractmethod
    def to_json(self) -> dict:
        """Convert the run state to JSON-serializable dictionary"""
        pass

    @classmethod
    @abstractmethod
    def from_json(cls, data: dict) -> 'BaseRunStateDict':
        """Create instance from JSON data"""
        pass

    def to_json_file(self, file_path: str) -> None:
        """Save the run state to a JSON file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_json(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json_file(cls, file_path: str) -> 'BaseRunStateDict':
        """Load instance from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_json(data)

