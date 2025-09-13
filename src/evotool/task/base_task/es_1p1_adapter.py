import abc
from typing import List
from abc import abstractmethod

from .base_evaluator import Solution, EvaluationResult
from .base_adapter import BaseAdapter



class Es1p1Adapter(BaseAdapter):
    """ES(1+1) Adapter"""
    def __init__(self, task_info: dict):
        BaseAdapter.__init__(self, task_info)

    def make_init_sol(self) -> Solution:
        return self.make_init_sol_wo_other_info()

    @abstractmethod
    def get_prompt(self, best_sol:Solution) -> List[dict]:
        raise NotImplementedError()

