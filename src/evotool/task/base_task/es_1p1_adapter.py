import abc
from typing import List
from abc import abstractmethod

from .base_evaluator import Solution, EvaluationResult



class Es1p1Adapter(abc.ABC):
    """ES(1+1) Adapter"""
    def __init__(self, task_info: dict):
        self.task_info = task_info

    @abstractmethod
    def make_init_sol(self) -> Solution:
        """Create initial solution from task info."""
        raise NotImplementedError()

    @abstractmethod
    def get_prompt(self, best_sol:Solution) -> List[dict]:
        raise NotImplementedError()

    @abstractmethod
    def parse_response(self, response_str: str) -> Solution:
        raise NotImplementedError()