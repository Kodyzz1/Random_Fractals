# noise_utils.py

import numpy as np
import opensimplex
import cupy as cp
import logging

def generate_perlin_noise_cpu(x_gpu, y_gpu, z_gpu, octaves=6, persistence=0.5, lacunarity=2.0, scale=100.0, seed=0):
    """
    Generates OpenSimplex noise on the CPU for CuPy input arrays.

    Args:
        x_gpu (cupy.ndarray): CuPy array of X coordinates.
        y_gpu (cupy.ndarray): CuPy array of Y coordinates.
        z_gpu (float): Z coordinate (not directly used in OpenSimplex 2D).
        octaves (int): Number of octaves for noise (not directly used).
        persistence (float): Persistence value (not directly used).
        lacunarity (float): Lacunarity value (not directly used).
        scale (float): Scale of the noise.
        seed (int): Seed for the OpenSimplex generator.

    Returns:
        cupy.ndarray: CuPy array of OpenSimplex noise values.
    """
    try:
        x_cpu = cp.asnumpy(x_gpu)  # Convert CuPy arrays to NumPy
        y_cpu = cp.asnumpy(y_gpu)
        noise_values = np.zeros_like(x_cpu)  # Create a NumPy array to store the results
        opensimplex.seed(seed) # set the seed

        for i in range(x_cpu.shape[0]):
            for j in range(x_cpu.shape[1]):
                noise_values[i, j] = opensimplex.noise2(x_cpu[i, j] / scale, y_cpu[i, j] / scale) #corrected method call

        return cp.asarray(noise_values)  # Convert back to CuPy array

    except Exception as e:
        logging.error(f"Error in generate_perlin_noise_cpu: {e}")
        return cp.zeros_like(x_gpu)  # Return zeros on error