from evotool.tools.llm import HttpsApi
from evotool.evaluator import Evaluator
from ..base_config import BaseConfig
from typing import List

class EvoEngineerConfig(BaseConfig):
    def __init__(
            self, output_path,
            evaluator: Evaluator,
            running_llm: HttpsApi,
            verbose: bool = True
    ):
        super().__init__(output_path, verbose)
        self.evaluator = evaluator
        self.running_llm = running_llm