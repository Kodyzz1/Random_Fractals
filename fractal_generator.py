import image_renderer
import fractal_math  # Make sure to include the updated fractal_math.py
import time
import ffmpeg
import logging
import os
import cupy as cp
import random
import shutil


def choose_interesting_point(width, height, center_x, center_y, initial_zoom, max_iter, fractal_type, julia_c=None):
    """Chooses an interesting point based on iteration count variance."""

    preview_width = 200  # Increased for better refinement
    preview_height = 200
    x_coords = cp.linspace(-1, 1, preview_width) * initial_zoom + center_x
    y_coords = cp.linspace(-1, 1, preview_height) * initial_zoom + center_y
    c = x_coords[:, cp.newaxis] + 1j * y_coords[cp.newaxis, :]
    preview_max_iter = 50 # Keep calculation fast

    if fractal_type == "mandelbrot":
        iterations = fractal_math.mandelbrot_gpu(c, preview_max_iter)
        lower_bound = cp.percentile(iterations, 10)  # Wider range
        upper_bound = cp.percentile(iterations, 90)
        interesting_points = cp.where((iterations >= lower_bound) & (iterations <= upper_bound))
    elif fractal_type == "julia":
        z = c
        iterations = fractal_math.julia_set_gpu(julia_c, z, preview_max_iter)
        lower_bound = cp.percentile(iterations, 10)
        upper_bound = cp.percentile(iterations, 90)
        interesting_points = cp.where((iterations >= lower_bound) & (iterations <= upper_bound))

    elif fractal_type == "burning_ship":
        iterations = fractal_math.burning_ship_gpu(c, preview_max_iter)
        lower_bound = cp.percentile(iterations, 10)
        upper_bound = cp.percentile(iterations, 90)
        interesting_points = cp.where((iterations >= lower_bound) & (iterations <= upper_bound))
    else:
        raise ValueError(f"Invalid fractal type: {fractal_type}")

    if interesting_points[0].size == 0:
        logging.warning("No interesting points found (initial selection). Using a random point.")
        if fractal_type == "mandelbrot":
            return random.uniform(-1.5, 0.5), random.uniform(-1.0, 1.0)
        elif fractal_type == "julia":
            return random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)
        elif fractal_type == "burning_ship":
            return random.uniform(-2,1), random.uniform(-2, 0)


    # --- Local Variance Refinement ---
    variances = fractal_math.local_variance(iterations, window_size=5) # Use the function!

    # Get the variances at the interesting points.
    interesting_variances = variances[interesting_points]

    # Find the index of the *maximum* variance among the interesting points.
    best_index = cp.argmax(interesting_variances)

    chosen_x_index = interesting_points[0][best_index]
    chosen_y_index = interesting_points[1][best_index]

    target_x = float(x_coords[chosen_x_index])
    target_y = float(y_coords[chosen_y_index])
    return target_x, target_y
def get_resolution():
    """Prompts the user to select a resolution."""
    print("Select a resolution:")
    print("1. 480p (640x480)")
    print("2. 720p (1280x720)")
    print("3. 1080p (1920x1080)")
    print("4. 1440p (2560x1440)")
    print("5. 2160p (3840x2160)")

    while True:
        choice = input("Enter your choice (1-5): ")
        if choice == '1':
            return 640, 480
        elif choice == '2':
            return 1280, 720
        elif choice == '3':
            return 1920, 1080
        elif choice == '4':
            return 2560, 1440
        elif choice == '5':
            return 3840, 2160
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

def get_integer_input(prompt, min_val=None, max_val=None, example=None):
    """Gets an integer input from the user with validation and optional example."""
    while True:
        try:
            if example:
                value = int(input(f"{prompt} (e.g., {example}): "))
            else:
                value = int(input(prompt))

            if min_val is not None and value < min_val:
                print(f"Please enter a value greater than or equal to {min_val}.")
            elif max_val is not None and value > max_val:
                print(f"Please enter a value less than or equal to {max_val}.")
            else:
                return value
        except ValueError:
            print("Invalid input. Please enter an integer.")

def get_float_input(prompt, min_val=None, max_val=None, example=None):
    """Gets a float input, with validation and optional example."""
    while True:
        try:
            if example:
                value = float(input(f"{prompt} (e.g., {example}): "))
            else:
                value = float(input(prompt))

            if min_val is not None and value < min_val:
                print(f"Please enter a value greater than or equal to {min_val}.")
            elif max_val is not None and value > max_val:
                print(f"Please enter a value less than or equal to {max_val}.")
            else:
                return value
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_fractal_type():
    """Prompts user to select a fractal type"""
    print("Select a fractal type")
    print("1. Mandelbrot")
    print("2. Julia")
    print("3. Burning Ship")

    while True:
        choice = input("Enter your choice (1-3): ")
        if choice == '1':
            return "mandelbrot"
        elif choice == '2':
            return "julia"
        elif choice == '3':
            return "burning_ship"
        else:
            print("Invalid choice. Please enter a number between 1 and 3.")

def get_color_map():
    """Prompts the user to select a color map"""
    print("Select a color map:")
    print("1. Grayscale")
    print("2. Rainbow")

    while True:
        choice = input("Enter your choice (1-2): ")
        if choice == '1':
            return "grayscale"
        elif choice == '2':
            return "rainbow"
        else:
            print("Invalid choice. Please enter a number between 1 and 2.")

def generate_fractal_video_gpu(filename, total_frames, width, height, center_x, center_y, zoom_speed, max_iter, fps, fractal_type, color_map, noise_scale, noise_strength, noise_octaves, noise_persistence, noise_lacunarity, render_delay):
    """Generates a fractal video."""

    logging.info(f"Generating fractal video: {filename}")

    # --- Get the absolute path of the current working directory ---
    current_dir = os.path.abspath(".")  # Get absolute path
    logging.info(f"Current working directory: {current_dir}")

    initial_zoom = 2.0
    if fractal_type == "julia":
        julia_c = fractal_math.DEFAULT_JULIA_C
        target_x, target_y = choose_interesting_point(width, height, center_x, center_y, initial_zoom, 50, fractal_type, julia_c)
    else:
        target_x, target_y = choose_interesting_point(width, height, center_x, center_y, initial_zoom, 50, fractal_type)
    logging.info(f"Target point: ({target_x}, {target_y})")

    try:
        start_time = time.time()
        frame_num = 0
        zoom = initial_zoom
        while zoom > 0.0001 and frame_num < total_frames:
            # --- Center Adjustment ---
            center_x = center_x + (target_x - center_x) * zoom_speed / initial_zoom
            center_y = center_y + (target_y - center_y) * zoom_speed / initial_zoom

            # --- Use absolute paths for frame filenames ---
            frame_filename = os.path.join(current_dir, f"frame_{frame_num:04d}.png")  # Absolute path
            logging.info(f"Frame: {frame_num}, Zoom: {zoom}, Center: ({center_x}, {center_y}), Filename: {frame_filename}")

            image_renderer.render_fractal_frame_to_png(frame_filename, width, height, center_x, center_y, zoom, max_iter, fractal_type, color_map, noise_scale, noise_strength, noise_octaves, noise_persistence, noise_lacunarity)

            cp._default_memory_pool.free_all_blocks()

            percentage = (frame_num + 1) / total_frames * 100
            print(f"Frame Progress: {percentage:.2f}%", end="\r")
            elapsed_time = time.time() - start_time
            remaining_time = (elapsed_time / (frame_num + 1)) * (total_frames - frame_num - 1)
            logging.info(f"Frame {frame_num + 1}/{total_frames} rendered. Estimated time remaining: {remaining_time:.2f} seconds")

            time.sleep(render_delay)
            zoom = initial_zoom - (frame_num + 1) * zoom_speed
            frame_num += 1

        # --- Use absolute path for input to ffmpeg ---
        input_pattern = os.path.join(current_dir, 'frame_%04d.png')  # Absolute path
        (
            ffmpeg
            .input(input_pattern, format='image2', framerate=fps)
            .output(filename, crf=20, pix_fmt='yuv420p', y=None)
            .run()  #  Remove quiet=True for debugging if needed
        )
        logging.info(f"Video created: {filename}")
        print(f"\nVideo saved as {filename}")


    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")

    finally:
        # --- Clean up PNG frames (in a finally block) ---
        for i in range(frame_num):
            try:
                frame_path = os.path.join(current_dir, f"frame_{i:04d}.png")  # Absolute path
                os.remove(frame_path)
            except Exception as e:
                logging.error(f"Error deleting frame {i}: {e}")
        logging.info("Temporary PNG frames deleted.")

if __name__ == "__main__":
    logging.basicConfig(filename='fractal_generator.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Get user inputs ---
    print("Welcome to the Fractal Video Generator!")
    filename = input("Enter the output video filename (e.g., my_fractal.mp4): ")
    total_frames = get_integer_input("Enter the total number of frames: ", min_val=1)
    width, height = get_resolution()
    fractal_type = get_fractal_type()
    color_map = get_color_map()
    max_iter = get_integer_input("Enter the maximum number of iterations (higher = more detail, but slower): ", min_val=1, example=200)
    zoom_speed = get_float_input("Enter the zoom speed (lower = slower zoom, e.g., 0.01): ", min_val=0.0, example=0.02)
    fps = get_integer_input("Enter the frames per second for the video: ", min_val=1, example=30)
    render_delay = get_float_input("Enter the render delay in seconds between frames (optional, e.g., 0.05): ", example=0.05)

    # Default values for less commonly changed parameters
    center_x = -0.5
    center_y = 0.0
    noise_scale = 5.0
    noise_strength = 0.1
    noise_octaves = 6
    noise_persistence = 0.5
    noise_lacunarity = 2.0

    generate_fractal_video_gpu(filename, total_frames, width, height, center_x, center_y, zoom_speed, max_iter, fps, fractal_type, color_map, noise_scale, noise_strength, noise_octaves, noise_persistence, noise_lacunarity, render_delay)