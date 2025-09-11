import tempfile
import os
import shutil

from .evaluator import Evaluator
from ..base_task.base_evaluator import BaseEvaluator, EvaluationResult, TaskInfoMaker


class CudaTaskInfoMaker(TaskInfoMaker):
    @classmethod
    def make_task_info(
            cls,
            evaluator: Evaluator,
            gpu_type:str, cuda_version:str,
            org_py_code:str, func_py_code:str, cuda_code:str,
            **kwargs
    ) -> dict:
        task_info = {
            "gpu_type": gpu_type,
            "cuda_version": cuda_version,
            "org_py_code": org_py_code,
            "func_py_code": func_py_code,
            "cuda_code": cuda_code
        }
        LOCK_FILE = os.path.join(tempfile.gettempdir(), "evotool_cross_process.lock")
        shutil.rmtree(LOCK_FILE, ignore_errors=True)
        cuda_info_dict = evaluator.get_cuda_runtime_sandbox(
            func_py_code, cuda_code
        )
        info_dict = {
            "name": "baseline",
            "thought": "baseline",
            "code": cuda_code,
            "temp_str": cuda_info_dict["temp_str"],
            "runtime": cuda_info_dict["runtime"],
            "prof_string": cuda_info_dict["prof_string"],
            "compilation_error": False,
            "comparison_error": False
        }
        task_info["cuda_info"] = info_dict
        return task_info


class CudaEvaluator(BaseEvaluator):
    """CUDA optimization evaluator with built-in evaluation"""
    def __init__(
        self, task_info_dict: dict, temp_path
    ):
        super().__init__(task_info_dict)
        self.org_py_code = task_info_dict["org_py_code"]
        self.func_py_code = task_info_dict["func_py_code"]
        self.cuda_code = task_info_dict["cuda_code"]
        self.temp_path = temp_path

        self.evaluator = Evaluator(temp_path)
    
    # Evaluation methods
    def evaluate_code(self, candidate_code: str) -> EvaluationResult:
        """Evaluate CUDA kernel code using the original evaluator"""
        
        try:
            # Step 1: Evaluate CUDA code correctness
            cuda_comparison_result = self.evaluator.compare_func_cuda_sandbox(
                self.func_py_code,
                candidate_code
            )
            
            # Initialize additional_info structure similar to new_entry
            additional_info = {
                "code": candidate_code,
                "temp_str": cuda_comparison_result.get("temp_str"),
                "runtime": None,
                "prof_string": None,
                "compilation_error": cuda_comparison_result.get("compilation_error", False),
                "comparison_error": not cuda_comparison_result.get("correctness", False),
                "error_msg": cuda_comparison_result.get("error_msg", None)
            }
            
            # Step 2: If correct, measure runtime performance
            if cuda_comparison_result.get("correctness", False):
                cuda_runtime_result = self.evaluator.get_cuda_runtime_sandbox(
                    self.func_py_code,
                    candidate_code,
                    cuda_comparison_result.get("temp_str")
                )
                additional_info["runtime"] = cuda_runtime_result.get("runtime")
                additional_info["prof_string"] = cuda_runtime_result.get("prof_string")
                
                # Use runtime as score (lower is better for runtime optimization)
                score = -cuda_runtime_result.get("runtime")
                valid = True
                additional_info["error_msg"] = cuda_runtime_result.get("error_msg", None)
            else:
                score = None
                valid = False
            return EvaluationResult(
                valid=valid,
                score=score,
                additional_info=additional_info
            )
            
        except Exception as e:
            return EvaluationResult(
                valid=False,
                score=None,
                additional_info={
                    "code": candidate_code,
                    "temp_str": None,
                    "runtime": None,
                    "prof_string": None,
                    "compilation_error": True,
                    "comparison_error": True,
                    "error_msg": str(e),
                    "exception": True
                }
            )