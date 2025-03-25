import cupy as cp
import imageio.v3 as iio
import fractal_math
import logging
import os
import matplotlib.cm as cm
import numpy as np

def render_fractal_frame_to_png(filename: str, width: int, height: int, center_x: float, center_y: float, zoom: float, max_iter: int, fractal_type: str, color_map: str, noise_scale: float, noise_strength: float, noise_octaves: int, noise_persistence: float, noise_lacunarity: float):
    """
    Renders a fractal frame and saves it as a PNG file, with colormaps.
    """
    x_coords = cp.linspace(-1, 1, width) * zoom + center_x
    y_coords = cp.linspace(-1, 1, height) * zoom + center_y
    c = x_coords[:, cp.newaxis] + 1j * y_coords[cp.newaxis, :]

    if fractal_type == "mandelbrot":
        iterations = fractal_math.noisy_mandelbrot_gpu(c, max_iter, noise_scale, noise_strength, noise_octaves, noise_persistence, noise_lacunarity)
    elif fractal_type == "julia":
        z = c
        iterations = fractal_math.julia_set_gpu(fractal_math.DEFAULT_JULIA_C, z, max_iter)
    elif fractal_type == "burning_ship":
        iterations = fractal_math.burning_ship_gpu(c, max_iter)
    else:
        raise ValueError(f"Invalid fractal type: {fractal_type}")

    # --- Colormap Application ---
    # Normalize iteration counts (logarithmic scale for better distribution)
    iterations = cp.asnumpy(iterations)  # Convert to NumPy for matplotlib
    #avoid log 0 errors
    zero_mask = iterations == 0
    iterations = np.log(iterations + 1)  # Add 1 to avoid log(0)
    normalized_iterations = iterations / np.log(max_iter + 1)


    # Apply the chosen colormap
    cmap = cm.get_cmap(color_map)  # Get the colormap by name
    colors = cmap(normalized_iterations)  # Get RGBA values

    # Convert to uint8 and remove alpha channel (we only need RGB)
    colors = (colors[:, :, :3] * 255).astype(np.uint8)  # Remove alpha and scale

    colors[zero_mask] = [0, 0, 0] #set zero iterations to black.

    image_array = colors
    try:
        iio.imwrite(filename, image_array)
    except Exception as e:
        logging.error(f"Error writing PNG file {filename}: {e}")
        print(f"Error writing PNG file {filename}: {e}")
        raise  # Re-raise the exception to stop execution
    finally:
        del image_array
        del colors
        del iterations
        del c