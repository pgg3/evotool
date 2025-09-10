import dataclasses
from abc import ABC, abstractmethod


class EvaluationResult:
    def __init__(self, valid, score, additional_info):
        self.valid = valid
        self.score = score
        self.additional_info = additional_info

class Solution:
    def __init__(self, sol_string, other_info:dict=None, evaluation_res: EvaluationResult=None):
        self.sol_string = sol_string
        self.other_info = other_info
        self.evaluation_res = evaluation_res

class TaskInfoMaker(ABC):
    @classmethod
    @abstractmethod
    def make_task_info(cls, *args, **kwargs) -> dict:
        raise NotImplementedError()

class BaseEvaluator(ABC):
    def __init__(
        self, task_info:dict
    ):
        self.task_info = task_info
    
    # Evaluation methods
    @abstractmethod
    def evaluate_code(self, candidate_code: str) -> EvaluationResult:
        pass
