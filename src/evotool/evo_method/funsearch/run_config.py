from evotool.tools.llm import HttpsApi
from evotool.task.base_task import BaseEvaluator
from ..base_config import BaseConfig
from typing import Optional


class FunSearchConfig(BaseConfig):
    def __init__(
            self, 
            output_path,
            running_llm: HttpsApi,
            evaluator: BaseEvaluator,
            adapter,  # Will be FunSearchAdapter, but avoiding circular import
            max_sample_nums: Optional[int] = 200,
            num_islands: int = 10,
            max_population_size: int = 1000,
            num_samplers: int = 5,
            num_evaluators: int = 5,
            programs_per_prompt: int = 2,
            verbose: bool = True
    ):
        super().__init__(output_path, verbose)
        self.evaluator = evaluator
        self.running_llm = running_llm
        self.adapter = adapter
        self.max_sample_nums = max_sample_nums
        self.num_islands = num_islands
        self.max_population_size = max_population_size
        self.num_samplers = num_samplers
        self.num_evaluators = num_evaluators
        self.programs_per_prompt = programs_per_prompt