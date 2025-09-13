from evotool.task.base_task import BaseEvaluator, EvaluationResult


class PythonEvaluator(BaseEvaluator):
    """Simple Python code evaluator - users override evaluate_code method."""
    
    def __init__(self, task_info: dict, timeout_seconds: float = 30.0):
        """Initialize evaluator with task info.
        
        Args:
            task_info: Task information dictionary
            timeout_seconds: Execution timeout
        """
        super().__init__(task_info)
        self.timeout_seconds = timeout_seconds
    
    def evaluate_code(self, candidate_code: str) -> EvaluationResult:
        """Evaluate Python code.
        
        Users should override this method to implement their evaluation logic.
        Default implementation just returns success.
        
        Args:
            candidate_code: Python code to evaluate
            
        Returns:
            EvaluationResult with score and additional info
        """
        return EvaluationResult(
            valid=True,
            score=1.0,
            additional_info={'message': 'Default evaluation - please override evaluate_code method'}
        )