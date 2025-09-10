from typing import List
from evotool.task.python_task import FunSearchPythonAdapter, Solution


class FunSearchFuncApproxAdapter(FunSearchPythonAdapter):
    """FunSearch Adapter for function approximation tasks."""
    
    def make_init_sols(self) -> List[Solution]:
        """Create initial diverse solutions for function approximation."""
        solutions = []
        
        # Solution 1: Linear regression
        linear_code = '''def approximate(x):
    """Linear approximation."""
    import numpy as np
    
    x_mean = np.mean(x_train)
    y_mean = np.mean(y_train)
    
    numerator = np.sum((x_train - x_mean) * (y_train - y_mean))
    denominator = np.sum((x_train - x_mean) ** 2)
    
    if denominator == 0:
        slope = 0
    else:
        slope = numerator / denominator
    
    intercept = y_mean - slope * x_mean
    
    return slope * x + intercept'''
        
        solutions.append(Solution(linear_code, other_info={'generation': 0, 'method': 'linear'}))
        
        # Solution 2: Polynomial regression (degree 2)
        poly_code = '''def approximate(x):
    """Quadratic polynomial approximation."""
    import numpy as np
    
    # Create design matrix for polynomial features
    X = np.column_stack([np.ones(len(x_train)), x_train, x_train**2])
    
    # Solve normal equations: X.T @ X @ coeffs = X.T @ y
    coeffs = np.linalg.lstsq(X, y_train, rcond=None)[0]
    
    # Make predictions
    X_pred = np.column_stack([np.ones(len(x)), x, x**2])
    return X_pred @ coeffs'''
        
        solutions.append(Solution(poly_code, other_info={'generation': 0, 'method': 'polynomial'}))
        
        # Solution 3: Moving average approximation
        moving_avg_code = '''def approximate(x):
    """Moving average approximation."""
    import numpy as np
    
    y_pred = np.zeros_like(x)
    
    for i, xi in enumerate(x):
        # Find nearby training points
        distances = np.abs(x_train - xi)
        weights = 1.0 / (distances + 1e-8)
        
        # Weighted average
        y_pred[i] = np.sum(weights * y_train) / np.sum(weights)
    
    return y_pred'''
        
        solutions.append(Solution(moving_avg_code, other_info={'generation': 0, 'method': 'moving_average'}))
        
        return solutions
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for function approximation task."""
        return """You are an expert Python programmer specializing in function approximation algorithms.

Task Requirements:
- Function must be named 'approximate' and take parameter 'x' (numpy array)
- Use training data 'x_train' and 'y_train' available in the namespace
- Return predictions as numpy array for the input 'x'
- Optimize for R² score (coefficient of determination)

Guidelines:
- Analyze and combine strengths from multiple approaches
- Focus on mathematical synthesis: polynomial regression, spline interpolation, kernel methods, etc.
- Use numpy and math libraries for numerical computations
- Ensure numerical stability and handle edge cases
- Consider advanced techniques: regularization, ensemble methods, adaptive algorithms
- Vectorize operations for efficiency

Create a novel hybrid approach that synthesizes the best aspects of the provided solutions."""