# Function Approximation Tutorial

This tutorial demonstrates how to use the `func_approx` module from evotool to solve function approximation problems using various evolutionary algorithms.

## Overview

The `func_approx` module provides:
- Data generators for common function types (polynomial, sine wave, exponential decay)
- An evaluator that scores candidate functions based on their approximation quality
- Adapters for different evolutionary algorithms (ES, FunSearch, EOH, EvoEngineer)

## Basic Usage

### 1. Generate Training Data

First, let's generate some noisy data to approximate:

```python
from evotool.task.python_task.func_approx import generate_noisy_polynomial, generate_sine_wave

# Generate polynomial data: y = x³ - 2x² + x + 1 + noise
x_data, y_noisy, y_true = generate_noisy_polynomial(n_points=50, noise_level=0.1, seed=42)

# Or generate sine wave data
# x_data, y_noisy, y_true = generate_sine_wave(n_points=50, freq=2.0, noise_level=0.05)
```

### 2. Set Up the Evaluator

The evaluator expects your code to define an `approximate` function:

```python
from evotool.task.python_task.func_approx import FuncApproxEvaluator

evaluator = FuncApproxEvaluator(x_data, y_noisy, y_true, timeout_seconds=30.0)
```

### 3. Define a Candidate Solution

Your function must be named `approximate` and take the x data as input:

```python
candidate_code = """
def approximate(x):
    # Simple polynomial approximation
    return x**3 - 2*x**2 + x + 1
"""

# Test the candidate
result = evaluator.evaluate_code(candidate_code)
print(f"Valid: {result.valid}")
print(f"Score (R²): {result.score:.4f}")
print(f"MSE: {result.additional_info_dict['mse']:.4f}")
print(f"True R²: {result.additional_info_dict.get('true_r2', 'N/A')}")
```

## Complete Example with Evolution Strategy

Here's a complete example using the ES (1+1) adapter:

```python
import numpy as np
from evotool.task.python_task.func_approx import (
    generate_noisy_polynomial,
    FuncApproxEvaluator,
    Es1p1FuncApproxAdapter
)

# 1. Generate training data
x_data, y_noisy, y_true = generate_noisy_polynomial(n_points=50, noise_level=0.15, seed=123)

# 2. Create evaluator
evaluator = FuncApproxEvaluator(x_data, y_noisy, y_true)

# 3. Initial candidate (simple linear function)
initial_code = '''
def approximate(x):
    # Linear approximation: y = ax + b
    a = 1.0
    b = 0.0
    return a * x + b
'''

# 4. Set up ES adapter
es_adapter = Es1p1FuncApproxAdapter(
    population_size=1,
    evaluator=evaluator,
    max_generations=100
)

# 5. Run evolution
best_solution = es_adapter.run(initial_code)

print("Best solution found:")
print(best_solution.code)
print(f"Final score: {best_solution.fitness:.4f}")
```

## Advanced Example with Custom Data

You can also create your own custom function to approximate:

```python
from evotool.task.python_task.func_approx import generate_custom_function, FuncApproxEvaluator

# Define a complex custom function
def complex_function(x):
    return np.sin(x) * np.exp(-0.1 * x) + 0.5 * x

# Generate data
x_data, y_noisy, y_true = generate_custom_function(
    func=complex_function,
    x_range=(0, 10),
    n_points=100,
    noise_level=0.1,
    seed=42
)

# Create evaluator
evaluator = FuncApproxEvaluator(x_data, y_noisy, y_true)

# Test a candidate solution
candidate_code = """
import math

def approximate(x):
    # Try to approximate: sin(x) * exp(-0.1*x) + 0.5*x
    return np.sin(x) * np.exp(-0.1 * x) + 0.5 * x
"""

result = evaluator.evaluate_code(candidate_code)
print(f"Approximation quality: R² = {result.score:.4f}")
```

## Function Requirements

Your `approximate` function must:

1. **Be named exactly `approximate`**
2. **Take one parameter** (the x data array)
3. **Return a numpy array** with the same shape as the input
4. **Use only allowed imports**: `numpy` (as `np`) and `math` are available
5. **Complete within the timeout** (default 30 seconds)

### Valid Function Examples:

```python
# Polynomial approximation
def approximate(x):
    return 2*x**3 - x**2 + 3*x + 1

# Trigonometric approximation  
def approximate(x):
    return np.sin(2*x) + 0.5*np.cos(x)

# Exponential approximation
def approximate(x):
    return 2 * np.exp(-0.5 * x)

# Piecewise approximation
def approximate(x):
    result = np.zeros_like(x)
    mask1 = x < 0
    mask2 = x >= 0
    result[mask1] = x[mask1]**2
    result[mask2] = np.sqrt(np.abs(x[mask2]))
    return result
```

## Evaluation Metrics

The evaluator provides several metrics:

- **score**: R-squared value (coefficient of determination) - higher is better
- **mse**: Mean Squared Error against noisy data
- **mae**: Mean Absolute Error against noisy data  
- **r2**: R-squared against noisy data
- **true_mse**: MSE against true function (if available)
- **true_mae**: MAE against true function (if available)
- **true_r2**: R-squared against true function (if available)

## Available Evolutionary Algorithms

The module includes adapters for various evolutionary algorithms:

- `Es1p1FuncApproxAdapter`: (1+1) Evolution Strategy
- `FunSearchFuncApproxAdapter`: FunSearch algorithm
- `EohFuncApproxAdapter`: Evolution of Heuristics
- `EvoEngineerFuncApproxAdapter`: EvoEngineer algorithm

Each adapter has its own parameters and can be configured based on your specific needs.

## Tips for Better Results

1. **Start simple**: Begin with basic polynomial or trigonometric functions
2. **Use domain knowledge**: If you know the function type, incorporate that into your approximation
3. **Handle edge cases**: Ensure your function works for all x values in the domain
4. **Monitor convergence**: Track the score over generations to see if evolution is progressing
5. **Experiment with noise levels**: Higher noise makes the problem more challenging but more realistic