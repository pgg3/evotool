import os
from abc import abstractmethod, ABC
from typing import List
from evotool.task.base_task import Solution

from .base_config import BaseConfig
from .base_run_state_dict import BaseRunStateDict

class Method(ABC):
    def __init__(self, config:BaseConfig):
        self.config = config

    def verbose_info(self, message:str):
        if self.config.verbose:
            print(message)
    
    def verbose_title(self, text: str, total_width: int = 60):
        """Display a centered title with equal signs above and below"""
        if self.config.verbose:
            print("=" * total_width)
            print(text.center(total_width))
            print("=" * total_width)
    
    def verbose_stage(self, text: str, total_width: int = 60):
        """Display a stage separator with dashes"""
        if self.config.verbose:
            print("-" * total_width)
            print(text.center(total_width))
            print("-" * total_width)

    def verbose_gen(self, text: str, total_width: int = 60):
        """Display text centered with dashes on both sides"""
        if self.config.verbose:
            padding = (total_width - len(text)) // 2
            left_dashes = "-" * padding
            right_dashes = "-" * (total_width - len(text) - padding)
            print(left_dashes + text + right_dashes)

    @staticmethod
    def _get_best_valid_sol(sol_list: List[Solution]):
        valid_sols = []
        for sol in sol_list:
            if sol.evaluation_res is not None:
                if sol.evaluation_res.valid:
                    valid_sols.append(sol)

        # Return the kernel with minimum runtime
        best_kernel = max(valid_sols, key=lambda x: x.evaluation_res.score)
        return best_kernel

    def _save_run_state(self, run_state_dict: BaseRunStateDict):
        """Save run state to file"""
        run_state_dict.to_json_file(os.path.join(self.config.output_path, "run_state.json"))