import numpy as np
from typing import Tuple


def generate_noisy_polynomial(n_points: int = 100, noise_level: float = 0.1, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate noisy polynomial data for function approximation.
    
    Args:
        n_points: Number of data points
        noise_level: Standard deviation of noise
        seed: Random seed for reproducibility
        
    Returns:
        x: Input values
        y_noisy: Noisy target values  
        y_true: True function values
    """
    np.random.seed(seed)
    x = np.linspace(-2, 2, n_points)
    y_true = x**3 - 2*x**2 + x + 1
    noise = np.random.normal(0, noise_level, n_points)
    y_noisy = y_true + noise
    
    return x, y_noisy, y_true


def generate_sine_wave(n_points: int = 100, freq: float = 2.0, noise_level: float = 0.05, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate noisy sine wave data for function approximation.
    
    Args:
        n_points: Number of data points
        freq: Frequency of sine wave
        noise_level: Standard deviation of noise
        seed: Random seed for reproducibility
        
    Returns:
        x: Input values
        y_noisy: Noisy target values
        y_true: True function values
    """
    np.random.seed(seed)
    x = np.linspace(0, 4*np.pi, n_points)
    y_true = np.sin(freq * x)
    noise = np.random.normal(0, noise_level, n_points)
    y_noisy = y_true + noise
    
    return x, y_noisy, y_true


def generate_exponential_decay(n_points: int = 100, decay_rate: float = 0.5, noise_level: float = 0.02, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate noisy exponential decay data for function approximation.
    
    Args:
        n_points: Number of data points
        decay_rate: Decay rate parameter
        noise_level: Standard deviation of noise
        seed: Random seed for reproducibility
        
    Returns:
        x: Input values
        y_noisy: Noisy target values
        y_true: True function values
    """
    np.random.seed(seed)
    x = np.linspace(0, 5, n_points)
    y_true = np.exp(-decay_rate * x)
    noise = np.random.normal(0, noise_level, n_points)
    y_noisy = y_true + noise
    
    return x, y_noisy, y_true


def generate_custom_function(func, x_range: Tuple[float, float], n_points: int = 100, noise_level: float = 0.1, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate noisy data for custom function.
    
    Args:
        func: Function to generate data from
        x_range: Range of x values (min, max)
        n_points: Number of data points
        noise_level: Standard deviation of noise
        seed: Random seed for reproducibility
        
    Returns:
        x: Input values
        y_noisy: Noisy target values
        y_true: True function values
    """
    np.random.seed(seed)
    x = np.linspace(x_range[0], x_range[1], n_points)
    y_true = func(x)
    noise = np.random.normal(0, noise_level, n_points)
    y_noisy = y_true + noise
    
    return x, y_noisy, y_true