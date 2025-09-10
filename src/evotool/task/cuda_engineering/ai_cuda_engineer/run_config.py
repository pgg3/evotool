from evotool.tools.llm import HttpsApi
from ..evaluator import Evaluator
from evotool.evo_method.base_config import BaseConfig
from typing import List
class AiCudaEngineerConfig(BaseConfig):
    def __init__(
            self,
            task_info: dict,
            output_path,
            evaluator: Evaluator,
            conversion_llm: HttpsApi,
            translation_llm: HttpsApi,
            evo_llm_list: List[HttpsApi],
            embedding_llm: HttpsApi,
            rag_llm: HttpsApi,
            conversion_retry: int=10
    ):
        super().__init__(task_info, output_path)
        self.evaluator = evaluator
        self.conversion_retry = conversion_retry
        self.conversion_llm = conversion_llm
        self.translation_llm = translation_llm
        self.evo_llm_list = evo_llm_list
        self.embedding_llm = embedding_llm
        self.rag_llm = rag_llm