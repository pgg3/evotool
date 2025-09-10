from evotool.task.python_task import Es1p1PythonAdapter, Solution


class Es1p1FuncApproxAdapter(Es1p1PythonAdapter):
    """ES(1+1) Adapter for function approximation tasks."""
    
    def make_init_sol(self) -> Solution:
        """Create initial solution for function approximation."""
        initial_code = '''def approximate(x):
    """Simple linear approximation as starting point."""
    import numpy as np
    
    # Simple linear regression
    x_mean = np.mean(x_train)
    y_mean = np.mean(y_train)
    
    # Calculate slope
    numerator = np.sum((x_train - x_mean) * (y_train - y_mean))
    denominator = np.sum((x_train - x_mean) ** 2)
    
    if denominator == 0:
        slope = 0
    else:
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

Return only the improved function implementation."""