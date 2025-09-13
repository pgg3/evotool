from abc import ABC, abstractmethod
from typing import List
from .base_evaluator import Solution
from .base_adapter import BaseAdapter



class FunSearchAdapter(BaseAdapter):
    """Base adapter for FunSearch algorithm"""
    
    def __init__(self, task_info: dict):
        BaseAdapter.__init__(self, task_info)

    def make_init_sol(self) -> Solution:
        return self.make_init_sol_wo_other_info()
    
    @abstractmethod
    def get_prompt(self, solutions: List[Solution]) -> List[dict]:
        """Generate prompt based on multiple solutions"""
        pass
