from abc import abstractmethod
from evotool.task import Solution, EvaluationResult

def _make_task_description(operation_name: str, GPU_TYPE: str, CUDA_VER: str) -> str:
    return f"""You are a Machine Learning Engineer trying to reduce the runtime of a {operation_name} kernel in CUDA. 
Make sure the kernel returns the correct result. Do not use any alternative precision that could result in an incorrect result. 
The kernel will be run on a {GPU_TYPE} GPU with CUDA version {CUDA_VER}.

"""


class CudaAdapter:
    """Cuda Adapter"""
    def __init__(self, task_info: dict):
        self.task_info = task_info

    def _get_base_task_description(self) -> str:
        """Get the base task description using task info"""
        operation_name = self.task_info.get("operation_name", "operation")
        gpu_type = self.task_info.get("gpu_type", "RTX 4090")
        cuda_version = self.task_info.get("cuda_version", "12.4.1")

        return _make_task_description(
            operation_name=operation_name,
            GPU_TYPE=gpu_type,
            CUDA_VER=cuda_version
        )

    def make_init_sol_wo_other_info(self) -> Solution:
        init_sol = Solution(self.task_info["cuda_code"])
        evaluation_res = EvaluationResult(
            valid=True,
            score=-self.task_info["cuda_info"]["runtime"],
            additional_info=dict()
        )
        init_sol.evaluation_res = evaluation_res
        return init_sol