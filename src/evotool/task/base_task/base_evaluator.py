import dataclasses
from abc import ABC, abstractmethod


class EvaluationResult:
    def __init__(self, valid, score, additional_info_dict):
        self.valid = valid
        self.score = score
        self.additional_info = additional_info_dict

class Solution:
    def __init__(self, sol_string, other_info:dict=None, evaluation_res: EvaluationResult=None):
        self.sol_string = sol_string
        self.other_info = other_info
        self.evaluation_res = evaluation_res

class BaseEvaluator(ABC):
    def __init__(
        self, timeout_seconds: float = 30.0
    ):
        self.timeout_seconds = timeout_seconds
    
    # Evaluation methods
    @abstractmethod
    def evaluate_code(self, candidate_code: str) -> EvaluationResult:
        pass
