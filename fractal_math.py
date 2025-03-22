import cupy as cp
from cupyx.scipy.ndimage import convolve  # More specific import
import noise_utils

DEFAULT_JULIA_C = -0.8 + 0.156j  # Good default Julia constant

def mandelbrot_gpu(c: cp.ndarray, max_iter: int) -> cp.ndarray:
    """
    Calculates the Mandelbrot set using GPU operations.

    Args:
        c (cupy.ndarray): Complex coordinates.
        max_iter (int): Maximum iterations.

    Returns:
        cupy.ndarray: Iteration counts.
    """
    z = cp.zeros_like(c, dtype=cp.complex128)
    iterations = cp.zeros_like(c, dtype=cp.int32)
    mask = cp.ones_like(c, dtype=cp.bool_)

    for i in range(max_iter):
        z[mask] = z[mask] * z[mask] + c[mask]
        mask[cp.abs(z) > 2] = False
        iterations[mask] = i
        if cp.any(cp.isnan(z)) or cp.any(cp.isinf(z)):
            logging.warning("NaN or inf detected in mandelbrot calculation!")
            break

    return iterations

def julia_set_gpu(c: cp.ndarray, z: cp.ndarray, max_iter: int) -> cp.ndarray:
    """
    Calculates the Julia set using GPU operations.

    Args:
        c (cupy.ndarray): Complex constant for the Julia set.
        z (cupy.ndarray): Complex initial values (typically the coordinates).
        max_iter (int): Maximum iterations.

    Returns:
        cupy.ndarray: Iteration counts.
    """
    iterations = cp.zeros_like(z, dtype=cp.int32)
    mask = cp.ones_like(z, dtype=cp.bool_)

    for i in range(max_iter):
        z[mask] = z[mask] * z[mask] + c[mask]
        mask[cp.abs(z) > 2] = False
        iterations[mask] = i
        if cp.any(cp.isnan(z)) or cp.any(cp.isinf(z)):
            logging.warning("NaN or inf detected in julia calculation!")
            break

    return iterations

def burning_ship_gpu(c: cp.ndarray, max_iter: int) -> cp.ndarray:
    """
    Calculates the Burning Ship fractal using GPU operations.

    Args:
        c (cupy.ndarray): Complex coordinates.
        max_iter (int): Maximum iterations.

    Returns:
        cupy.ndarray: Iteration counts.
    """
    z = cp.zeros_like(c, dtype=cp.complex128)
    iterations = cp.zeros_like(c, dtype=cp.int32)
    mask = cp.ones_like(c, dtype=cp.bool_)

    for i in range(max_iter):
         z[mask] = (cp.abs(cp.real(z[mask])) + 1j * cp.abs(cp.imag(z[mask])))**2 + c[mask]
         mask[cp.abs(z) > 2] = False
         iterations[mask] = i
         if cp.any(cp.isnan(z)) or cp.any(cp.isinf(z)):
            logging.warning("NaN or inf detected in burning ship calculation!")
            break

    return iterations

def noisy_mandelbrot_gpu(c: cp.ndarray, max_iter: int, noise_scale: float, noise_strength: float, noise_octaves: int, noise_persistence: float, noise_lacunarity: float, seed: int = 0) -> cp.ndarray:
    """
    Calculates the Mandelbrot set with Perlin noise applied.

    Args:
        c (cupy.ndarray): Complex coordinates.
        max_iter (int): Maximum iterations.
        noise_scale (float): Scale of the noise.
        noise_strength (float): Strength of the noise.
        noise_octaves (int): Number of noise octaves.
        noise_persistence (float): Noise persistence.
        noise_lacunarity (float): Noise lacunarity.
        seed (int): Random seed.

    Returns:
        cupy.ndarray: Iteration counts with noise.
    """
    z = cp.zeros_like(c, dtype=cp.complex128)
    iterations = cp.zeros_like(c, dtype=cp.int32)
    mask = cp.ones_like(c, dtype=cp.bool_)

    for i in range(max_iter):
        z[mask] = z[mask] * z[mask] + c[mask]
        mask[cp.abs(z) > 2] = False
        iterations[mask] = i
        if cp.any(cp.isnan(z)) or cp.any(cp.isinf(z)):
            logging.warning("NaN or inf detected in noisy mandelbrot calculation!")
            break

    noise_values = noise_utils.generate_perlin_noise_cpu(cp.real(c), cp.imag(c), octaves=noise_octaves, persistence=noise_persistence, lacunarity=noise_lacunarity, scale=noise_scale, seed=seed)
    iterations = cp.clip(iterations + (noise_values * noise_strength).astype(cp.int32), 0, max_iter)

    return iterations


def local_variance(iterations: cp.ndarray, window_size: int = 3) -> cp.ndarray:
    """Calculates the local variance of an array using convolution.

    Args:
        iterations: A 2D CuPy array of iteration counts.
        window_size: The size of the square window (e.g., 3 for a 3x3 window).

    Returns:
        A CuPy array of the same shape as iterations, containing the local variance.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")

    # Create a square window of ones.
    window = cp.ones((window_size, window_size)) / (window_size * window_size)

    # Calculate the local mean (E[X]).
    mean = convolve(iterations, window, mode='constant', cval=0.0)

    # Calculate the mean of the squares (E[X^2]).
    squared_iterations = iterations * iterations
    mean_of_squares = convolve(squared_iterations, window, mode='constant', cval=0.0)

    # Calculate the variance (E[X^2] - (E[X])^2).
    variance = mean_of_squares - mean * mean
    return variance