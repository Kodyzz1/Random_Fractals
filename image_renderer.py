import cupy as cp
import imageio.v3 as iio  # Use imageio v3 API
import fractal_math
import logging

def render_fractal_frame_to_png(filename: str, width: int, height: int, center_x: float, center_y: float, zoom: float, max_iter: int, fractal_type: str, color_map: str, noise_scale: float, noise_strength: float, noise_octaves: int, noise_persistence: float, noise_lacunarity: float):
    """
    Renders a fractal frame and saves it as a PNG file.

    Args:
        filename (str): Output PNG filename.
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
    x_coords = cp.linspace(-1, 1, width) * zoom + center_x
    y_coords = cp.linspace(-1, 1, height) * zoom + center_y
    c = x_coords[:, cp.newaxis] + 1j * y_coords[cp.newaxis, :]

    if fractal_type == "mandelbrot":
        iterations = fractal_math.noisy_mandelbrot_gpu(c, max_iter, noise_scale, noise_strength, noise_octaves, noise_persistence, noise_lacunarity)
    elif fractal_type == "julia":
        z = c
        iterations = fractal_math.julia_set_gpu(c, z, max_iter)
    elif fractal_type == "burning_ship":
        iterations = fractal_math.burning_ship_gpu(c, max_iter)
    else:
        raise ValueError(f"Invalid fractal type: {fractal_type}")

    if color_map == "grayscale":
       colors = (iterations % 256).astype(cp.uint8)
    elif color_map == "rainbow":
       colors = (iterations * 10 % 256).astype(cp.uint8) #example rainbow color map
    else:
       raise ValueError(f"Invalid color map: {color_map}")


    image_array = cp.stack([colors, colors, colors], axis=-1)  # Create RGB image
    image_array_cpu = cp.asnumpy(image_array)
    iio.imwrite(filename, image_array_cpu)

    del image_array  # Explicitly delete to free GPU memory
    del colors
    del iterations
    del c