import sys
import io
import traceback
import numpy as np
from evotool.task.python_task import PythonEvaluator
from evotool.task.base_task import EvaluationResult


class FuncApproxEvaluator(PythonEvaluator):
    """Evaluator for function approximation tasks."""
    
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray, y_true: np.ndarray = None, timeout_seconds: float = 30.0):
        """Initialize function approximation evaluator.
        
        Args:
            x_data: Input data points
            y_data: Target values (with noise)  
            y_true: True function values (optional, for comparison)
            timeout_seconds: Execution timeout
        """
        task_info = {
            'x_data': x_data,
            'y_data': y_data,
            'y_true': y_true
        }
        super().__init__(task_info, timeout_seconds)
    
    def evaluate_code(self, candidate_code: str) -> EvaluationResult:
        """Evaluate Python code for function approximation."""
        try:
            # Create namespace with common imports and training data
            import math
            namespace = {
                '__builtins__': {
                    'len': len, 'range': range, 'enumerate': enumerate,
                    'zip': zip, 'map': map, 'filter': filter,
                    'sum': sum, 'min': min, 'max': max, 'abs': abs,
                    'print': print, 'str': str, 'int': int, 'float': float,
                    'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
                    '__import__': __import__,
                },
                'np': np,
                'math': math,
                'x_train': self.task_info['x_data'],
                'y_train': self.task_info['y_data'],
            }
            
            # Execute the code
            exec(candidate_code, namespace)
            
            # Check if the required function exists
            if 'approximate' not in namespace:
                return EvaluationResult(
                    valid=False, 
                    score=0.0,
                    additional_info={'error': 'Function "approximate" not found in code'}
                )
            
            # Get the approximation function and data
            approximate_func = namespace['approximate']
            x_data = self.task_info['x_data']
            y_data = self.task_info['y_data']
            y_true = self.task_info['y_true']
            
            # Get predictions
            y_pred = approximate_func(x_data)
            
            # Ensure predictions are numpy array
            if not isinstance(y_pred, np.ndarray):
                y_pred = np.array(y_pred)
            
            # Check shape compatibility
            if y_pred.shape != y_data.shape:
                return EvaluationResult(
                    valid=False,
                    score=0.0, 
                    additional_info={'error': f'Shape mismatch: expected {y_data.shape}, got {y_pred.shape}'}
                )
            
            # Calculate metrics
            mse = np.mean((y_pred - y_data) ** 2)
            mae = np.mean(np.abs(y_pred - y_data))
            
            # Calculate R-squared
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Score is based on R-squared (higher is better)
            score = max(0, r2)
            
            additional_info = {
                'mse': float(mse),
                'mae': float(mae),
                'r2': float(r2),
                'predictions_shape': y_pred.shape,
            }
            
            # If true values available, also calculate true error
            if y_true is not None:
                true_mse = np.mean((y_pred - y_true) ** 2)
                true_mae = np.mean(np.abs(y_pred - y_true))
                
                ss_res_true = np.sum((y_true - y_pred) ** 2)
                ss_tot_true = np.sum((y_true - np.mean(y_true)) ** 2)
                true_r2 = 1 - (ss_res_true / ss_tot_true) if ss_tot_true != 0 else 0
                
                additional_info.update({
                    'true_mse': float(true_mse),
                    'true_mae': float(true_mae), 
                    'true_r2': float(true_r2)
                })
            
            return EvaluationResult(valid=True, score=score, additional_info=additional_info)
            
        except Exception as e:
            return EvaluationResult(
                valid=False,
                score=0.0,
                additional_info={
                    'error': f'Evaluation error: {str(e)}',
                    'traceback': traceback.format_exc()
                }
            )