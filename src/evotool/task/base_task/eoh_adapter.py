from abc import ABC, abstractmethod
from typing import List
from .base_evaluator import Solution


class EohAdapter(ABC):
    """Base adapter for EoH (Evolution of Heuristics) algorithm"""
    
    def __init__(self, task_info: dict):
        self.task_info = task_info

    @abstractmethod
    def _get_base_task_description(self) -> str:
        pass
    
    @abstractmethod
    def make_init_sol(self) -> Solution:
        """Create initial solution"""
        pass
    
    @abstractmethod
    def get_prompt_i1(self) -> List[dict]:
        """Generate initialization prompt (I1 operator)"""
        pass
    
    @abstractmethod
    def get_prompt_e1(self, selected_individuals: List[Solution]) -> List[dict]:
        """Generate E1 (crossover) prompt"""
        pass
    
    @abstractmethod
    def get_prompt_e2(self, selected_individuals: List[Solution]) -> List[dict]:
        """Generate E2 (guided crossover) prompt"""
        pass
    
    @abstractmethod
    def get_prompt_m1(self, individual: Solution) -> List[dict]:
        """Generate M1 (mutation) prompt"""
        pass
    
    @abstractmethod
    def get_prompt_m2(self, individual: Solution) -> List[dict]:
        """Generate M2 (parameter mutation) prompt"""
        pass
    
    @abstractmethod
    def parse_response(self, response_str: str) -> tuple[str, str]:
        """Parse LLM response to extract (code, algorithm) tuple"""
        pass