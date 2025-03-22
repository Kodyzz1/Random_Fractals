# svg_renderer.py

import cupy as cp
import fractal_math
import logging
import time

def render_fractal_frame_gpu(filename, width, height, center_x, center_y, zoom, max_iter, fractal_type="mandelbrot", color_map="grayscale", noise_scale=5.0, noise_strength=0.1, noise_octaves=6, noise_persistence=0.5, noise_lacunarity=2.0):
    """
    Renders a fractal frame as an SVG file.

    Args:
        filename (str): Output SVG filename.
        width (int): Frame width.
        height (int): Frame height.
        center_x (float): Center X coordinate.
        center_y (float): Center Y coordinate.
        zoom (float): Zoom factor.
        max_iter (int): Maximum iterations.
        fractal_type (str): Fractal type ("mandelbrot", "julia", "burning_ship").
        color_map (str): Color mapping function ("grayscale", "rainbow").
        noise_scale (float): Scale of the Perlin noise (for Mandelbrot).
        noise_strength (float): Strength of the noise effect (for Mandelbrot).
        noise_octaves (int): Octaves of noise (for Mandelbrot).
        noise_persistence (float): Persistence of noise (for Mandelbrot).
        noise_lacunarity (float): Lacunarity of noise (for Mandelbrot).
    """
    try:
        x_coords = cp.linspace(-1, 1, width) * zoom + center_x
        y_coords = cp.linspace(-1, 1, height) * zoom + center_y
        c = x_coords[:, cp.newaxis] + 1j * y_coords[cp.newaxis, :]  # Create complex coordinates

        if fractal_type == "mandelbrot":
            iterations = fractal_math.noisy_mandelbrot_gpu(c, max_iter, noise_scale, noise_strength, noise_octaves, noise_persistence, noise_lacunarity)
        elif fractal_type == "julia":
            z = c  # Julia set uses the complex coordinates as initial z values.
            iterations = fractal_math.julia_set_gpu(c, z, max_iter)
        elif fractal_type == "burning_ship":
            iterations = fractal_math.burning_ship_gpu(c, max_iter)
        else:
            raise ValueError(f"Invalid fractal type: {fractal_type}")

        # Color mapping
        if color_map == "grayscale":
            colors = (iterations % 256).astype(cp.int32)
        elif color_map == "rainbow":
            colors = (iterations * 10 % 256).astype(cp.int32) # example rainbow color map
        else:
            raise ValueError(f"Invalid color map: {color_map}")

        # SVG rendering
        start_time = time.time()
        with open(filename, 'w') as f:
            f.write(f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">\n')
            for y in range(height):
                for x in range(width):
                    color = int(colors[y, x])
                    f.write(f'<rect x="{x}" y="{y}" width="1" height="1" fill="rgb({color},{color},{color})" />\n')

                if y % (height // 10) == 0:
                    elapsed_time = time.time() - start_time
                    remaining_time = (elapsed_time / (y + 1)) * (height - y - 1)
                    logging.info(f"SVG Row Progress: {y/height * 100:.2f}%, Estimated time remaining: {remaining_time:.2f} seconds")

    except Exception as e:
        logging.error(f"Error in render_fractal_frame_gpu: {e}")
        # Add error handling or logging here if needed.
    finally:  # Ensure the closing tag is always written
        try:
            with open(filename, 'a') as f:  # Open in append mode.
                f.write('</svg>')
        except Exception as e:
            logging.error(f"Error writing closing SVG tag: {e}")