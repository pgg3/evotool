from evotool.task.python_task import EohPythonAdapter, Solution


class EohFuncApproxAdapter(EohPythonAdapter):
    """EOH Adapter for function approximation tasks.
    
    This adapter only needs to define the task-specific system prompt.
    All operator logic is handled by the parent EohPythonAdapter.
    """
    
    def make_init_sol(self) -> Solution:
        """Create initial solution for EOH."""
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
        
        return Solution(initial_code, other_info={'generation': 0, 'method': 'linear_regression'})
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for function approximation task."""
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
- Vectorize operations for efficiency

Output Format:
FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.
{algorithmic description and mathematical reasoning}
```python
function implementation code
```"""