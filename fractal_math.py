import cupy as cp
import noise_utils

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
        z[mask] = z[mask] * z[mask] + c[mask]  # Corrected line
        mask[cp.abs(z) > 2] = False
        iterations[mask] = i

    return iterations

def julia_set_gpu(c: cp.ndarray, z: cp.ndarray, max_iter: int) -> cp.ndarray:
    """
    Calculates the Julia set using GPU operations.

    Args:
        c (cupy.ndarray): Complex constant.
        z (cupy.ndarray): Complex initial values.
        max_iter (int): Maximum iterations.

    Returns:
        cupy.ndarray: Iteration counts.
    """
    iterations = cp.zeros_like(c, dtype=cp.int32)
    mask = cp.ones_like(c, dtype=cp.bool_)

    for i in range(max_iter):
        z[mask] = z[mask] * z[mask] + c[mask]
        mask[cp.abs(z) > 2] = False
        iterations[mask] = i

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
        z[mask] = (cp.abs(cp.real(z[mask])) + 1j * cp.abs(cp.imag(z[mask])))**2 + c[mask]  # Corrected line
        mask[cp.abs(z) > 2] = False
        iterations[mask] = i

    return iterations

def noisy_mandelbrot_gpu(c: cp.ndarray, max_iter: int, noise_scale: float, noise_strength: float, noise_octaves: int, noise_persistence: float, noise_lacunarity: float, seed: int = 0) -> cp.ndarray:
    """
    Calculates the Mandelbrot set with Perlin noise applied to the iteration count.

    Args:
        c (cupy.ndarray): Complex coordinates.
        max_iter (int): Maximum iterations.
        noise_scale (float): Scale of the noise.
        noise_strength (float): Strength of the noise effect.
        noise_octaves (int): Number of octaves for noise.
        noise_persistence (float): Persistence of noise.
        noise_lacunarity (float): Lacunarity of noise.
        seed (int): Seed for the OpenSimplex generator.

    Returns:
        cupy.ndarray: Iteration counts with noise applied.
    """
    z = cp.zeros_like(c, dtype=cp.complex128)
    iterations = cp.zeros_like(c, dtype=cp.int32)
    mask = cp.ones_like(c, dtype=cp.bool_)

    for i in range(max_iter):
        z[mask] = z[mask] * z[mask] + c[mask]
        mask[cp.abs(z) > 2] = False
        iterations[mask] = i


    noise_values = noise_utils.generate_perlin_noise_cpu(cp.real(c), cp.imag(c), octaves=noise_octaves, persistence=noise_persistence, lacunarity=noise_lacunarity, scale=noise_scale, seed=seed)
    iterations = cp.clip(iterations + (noise_values * noise_strength).astype(cp.int32), 0, max_iter)

    return iterations