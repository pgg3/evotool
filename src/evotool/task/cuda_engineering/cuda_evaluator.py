import time
from typing import Optional, List, Any, Tuple
from ..base_task.base_evaluator import BaseEvaluator, EvaluationResult
from evotool.evaluator import Evaluator

class CudaEvaluator(BaseEvaluator):
    """CUDA optimization evaluator with built-in evaluation"""
    def __init__(
        self, task_info_dict: dict, temp_path, timeout_seconds: float = 30.0
    ):
        super().__init__(timeout_seconds=timeout_seconds)
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
                "comparison_error": not cuda_comparison_result.get("correctness", False)
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
            else:
                score = None
                valid = False
            return EvaluationResult(
                valid=valid,
                score=score,
                additional_info_dict=additional_info
            )
            
        except Exception as e:
            return EvaluationResult(
                valid=False,
                score=None,
                additional_info_dict={
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