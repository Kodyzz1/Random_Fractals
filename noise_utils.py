import numpy as np
import opensimplex
import cupy as cp
import logging

def generate_perlin_noise_cpu(x_gpu: cp.ndarray, y_gpu: cp.ndarray, octaves:int = 6, persistence:float = 0.5, lacunarity:float = 2.0, scale: float = 100.0, seed: int = 0) -> cp.ndarray:
    """
    Generates OpenSimplex noise on the CPU for CuPy input arrays.

    Args:
        x_gpu (cupy.ndarray): CuPy array of X coordinates.
        y_gpu (cupy.ndarray): CuPy array of Y coordinates.
        octaves (int): Number of octaves for noise (not directly used with noise2).
        persistence (float): Persistence value (not directly used with noise2).
        lacunarity (float): Lacunarity value (not directly used with noise2).
        scale (float): Scale of the noise.
        seed (int): Seed for the OpenSimplex generator.

    Returns:
        cupy.ndarray: CuPy array of OpenSimplex noise values.
    """
    try:
        x_cpu = cp.asnumpy(x_gpu)  # Convert CuPy arrays to NumPy
        y_cpu = cp.asnumpy(y_gpu)
        noise_values = np.zeros_like(x_cpu, dtype=np.float64)  # Create a NumPy array to store the results
        opensimplex.seed(seed) # set the seed
        
        for i in range(x_cpu.shape[0]):
           for j in range(x_cpu.shape[1]):
                noise_values[i, j] = opensimplex.noise2(x_cpu[i, j] / scale, y_cpu[i, j] / scale)

        return cp.asarray(noise_values)  # Convert back to CuPy array

    except Exception as e:
        logging.error(f"Error in generate_perlin_noise_cpu: {e}")
        return cp.zeros_like(x_gpu)  # Return zeros on error