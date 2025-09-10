from abc import ABC, abstractmethod
from typing import List
from .base_evaluator import Solution


class FunSearchAdapter(ABC):
    """Base adapter for FunSearch algorithm"""
    
    def __init__(self, task_info: dict):
        self.task_info = task_info
    
    @abstractmethod
    def make_init_sol(self) -> Solution:
        """Create initial solution"""
        pass
    
    @abstractmethod
    def get_prompt(self, solutions: List[Solution]) -> List[dict]:
        """Generate prompt based on multiple solutions"""
        pass
    
    @abstractmethod
    def parse_response(self, response_str: str) -> str:
        """Parse LLM response to extract solution string"""
        pass