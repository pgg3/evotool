import abc
from abc import abstractmethod
from evotool.task import Solution


class BaseAdapter(abc.ABC):
    """Base Adapter"""
    def __init__(self, task_info: dict):
        self.task_info = task_info

    # Task-wise methods
    @abstractmethod
    def _get_base_task_description(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def make_init_sol_wo_other_info(self) -> Solution:
        """Create initial solution from task info without other_info."""
        raise NotImplementedError()


    # Method-wise methods
    @abstractmethod
    def make_init_sol(self) -> Solution:
        """Create initial solution from task info."""
        raise NotImplementedError()

    @abstractmethod
    def parse_response(self, response_str: str) -> Solution:
        raise NotImplementedError()