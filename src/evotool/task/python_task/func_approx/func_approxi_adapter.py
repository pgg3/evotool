from abc import ABC
from evotool.task import Solution


class FuncApproxBaseAdapter:
    """Base class for function approximation adapters with shared task description."""
    
    def make_init_sol_wo_other_info(self) -> Solution:
        """Create initial solution for function approximation."""
        initial_code = '''def approximate(x):
    """Linear regression as initial solution."""
    import numpy as np
    
    x_mean = np.mean(x_train)
    y_mean = np.mean(y_train)
    
    numerator = np.sum((x_train - x_mean) * (y_train - y_mean))
    denominator = np.sum((x_train - x_mean) ** 2)
    
    if denominator == 0:
        return np.full_like(x, y_mean)
    
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    return slope * x + intercept'''
        
        return Solution(initial_code)
    
    def _get_base_task_description(self) -> str:
        """Get shared task description for function approximation."""
        return """You are an expert Python programmer specializing in function approximation algorithms.

Task Requirements:
- Function must be named 'approximate' and take parameter 'x' (numpy array)
- Use training data 'x_train' and 'y_train' available in the namespace
- Return predictions as numpy array for the input 'x'
- Optimize for R² score (coefficient of determination)

Guidelines:
- Focus on mathematical approaches: polynomial regression, spline interpolation, kernel methods, etc.
- Use numpy and math libraries for numerical computations
- Ensure numerical stability and handle edge cases
- Consider regularization techniques to prevent overfitting
- Vectorize operations for efficiency"""