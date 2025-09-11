from evotool.tools.llm import HttpsApi
from evotool.task.base_task import BaseEvaluator, EohAdapter
from ..base_config import BaseConfig
from typing import Optional, Literal

class EohConfig(BaseConfig):
    def __init__(
            self,
            output_path,
            running_llm: HttpsApi,
            evaluator: BaseEvaluator,
            adapter: EohAdapter,
            max_generations: int = 10,
            max_sample_nums: int = 100,
            pop_size: int = 5,
            selection_num: int = 2,
            use_e2_operator: bool = True,
            use_m1_operator: bool = True,
            use_m2_operator: bool = True,
            num_samplers: int = 1,
            num_evaluators: int = 1,
            verbose: bool = True
    ):
        super().__init__(output_path, verbose)
        self.running_llm = running_llm
        self.evaluator = evaluator
        self.adapter = adapter
        self.max_generations = max_generations
        self.max_sample_nums = max_sample_nums
        self.pop_size = pop_size
        self.selection_num = selection_num
        self.use_e2_operator = use_e2_operator
        self.use_m1_operator = use_m1_operator
        self.use_m2_operator = use_m2_operator
        self.num_samplers = num_samplers
        self.num_evaluators = num_evaluators