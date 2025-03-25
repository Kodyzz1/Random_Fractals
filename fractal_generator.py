import image_renderer
import fractal_math
import time
import ffmpeg
import logging
import os
import cupy as cp
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter


def choose_interesting_point(width, height, center_x, center_y, zoom, max_iter, fractal_type, prev_target_x=None, prev_target_y=None, julia_c=None):
    """Chooses a point with balanced black/white, prioritizing proximity to the previous target."""

    preview_width = 200
    preview_height = 200
    # Use np.linspace (NumPy) instead of cp.linspace (CuPy)
    x_coords = np.linspace(-1, 1, preview_width) * zoom + center_x
    y_coords = np.linspace(-1, 1, preview_height) * zoom + center_y
    c = x_coords[:, np.newaxis] + 1j * y_coords[np.newaxis, :]
    preview_max_iter = 50

    if fractal_type == "mandelbrot":
        # Use the CPU version of Mandelbrot for the preview
        iterations = fractal_math.mandelbrot(c, preview_max_iter)  # Call the CPU version
    elif fractal_type == "julia":
        z = c
        # Use the CPU version of Julia for the preview
        iterations = fractal_math.julia_set(julia_c, z, preview_max_iter)  # Call the CPU version
    elif fractal_type == "burning_ship":
        # Use the CPU version of Burning Ship for the preview
        iterations = fractal_math.burning_ship(c, preview_max_iter) # Call the CPU version
    else:
        raise ValueError(f"Invalid fractal type: {fractal_type}")

    inside_mask = iterations >= preview_max_iter * 0.95
    window_size = 25
    inside_proportion = uniform_filter(inside_mask.astype(float), size=window_size, mode='constant')

    # --- ADJUSTED TOLERANCE AND ZOOM-DEPENDENT LOGIC ---
    tolerance = 0.25  # Increased tolerance significantly
    min_proportion = 0.2  # Minimum proportion of "inside" pixels
    max_proportion = 0.8  # Maximum proportion of "inside" pixels

    # Find points that meet the criteria
    interesting_points = np.where((inside_proportion >= min_proportion) & (inside_proportion <= max_proportion))


    if interesting_points[0].size == 0:
        logging.warning("No balanced points found. Using previous target.")
        if prev_target_x is not None and prev_target_y is not None:
            return prev_target_x, prev_target_y
        else:
            if fractal_type == "mandelbrot":
                return -0.75, 0.0
            elif fractal_type == "julia":
                return 0.0, 0.0
            elif fractal_type == "burning_ship":
                return -0.5, -0.5

    if prev_target_x is not None and prev_target_y is not None:
        distances = np.sqrt((x_coords[interesting_points[1]] - prev_target_x)**2 + (y_coords[interesting_points[0]] - prev_target_y)**2)
        closest_index = np.argmin(distances)
        chosen_x_index = interesting_points[1][closest_index]
        chosen_y_index = interesting_points[0][closest_index]
    else:
        choice_index = random.randint(0, interesting_points[0].size - 1)
        chosen_x_index = interesting_points[1][choice_index]
        chosen_y_index = interesting_points[0][choice_index]

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
    """Prompts the user to select a colormap, using friendly names."""

    # Dictionary mapping friendly names to Matplotlib colormap names
    colormaps = {
        "Fiery": "inferno",
        "Oceanic": "ocean",
        "Vibrant": "viridis",
        "Cool Warm": "coolwarm",
        "Twilight": "twilight_shifted",
        "Rainbow": "hsv",
        "Classic": "gnuplot2",
        "Pastel": "Pastel1",
        "Autumn": "autumn",
        "Grayscale": "gray",
    }

    print("Select a color map:")
    for i, (friendly_name, cmap_name) in enumerate(colormaps.items()):
        print(f"{i + 1}. {friendly_name}")

    while True:
        try:
            choice = int(input(f"Enter your choice (1-{len(colormaps)}): "))
            if 1 <= choice <= len(colormaps):
                # Get the friendly name (the key) from the dictionary
                selected_friendly_name = list(colormaps.keys())[choice - 1]
                # Use the friendly name to get the *actual* colormap name
                return colormaps[selected_friendly_name]
            else:
                print(f"Invalid choice.  Please enter a number between 1 and {len(colormaps)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def generate_single_fractal_image(filename, path_file, width, height, max_iter, fractal_type, color_map, noise_scale, noise_strength, noise_octaves, noise_persistence, noise_lacunarity):
    """Generates a single fractal image."""
    logging.info(f"Generating single fractal image: {filename}")

    try:
        path_data = np.load(path_file)
        path = path_data['path']
        if not path.size > 0:
            print("Error: Path file is empty.")
            return

        # Use the first point in the path for high zoom
        center_x, center_y, zoom = path[0]
        center_x = float(center_x)
        center_y = float(center_y)
        zoom = float(zoom)

        logging.info(f"Rendering single frame with Zoom: {zoom}, Center: ({center_x}, {center_y})")
        image_renderer.render_fractal_frame_to_png(filename, width, height, center_x, center_y, zoom, max_iter, fractal_type, color_map, noise_scale, noise_strength, noise_octaves, noise_persistence, noise_lacunarity)
        print(f"Single fractal image saved as {filename}")

    except FileNotFoundError:
        print(f"Error: Path file '{path_file}' not found.")
    except Exception as e:
        logging.error(f"Error generating single image: {e}")
        print(f"An error occurred: {e}")


def generate_fractal_video_gpu(filename, path_file, width, height, max_iter, fps, fractal_type, color_map, noise_scale, noise_strength, noise_octaves, noise_persistence, noise_lacunarity, render_delay, pan_speed=0.1, num_frames=None):
    """Generates a fractal video using a pre-calculated path."""

    logging.info(f"Generating fractal video: {filename}")

    # --- Load the pre-calculated path ---
    try:
        path_data = np.load(path_file)
        path = path_data['path']
        total_frames_in_path = len(path)  # Get total frames *from the path file*
        logging.info(f"Loaded path with {total_frames_in_path} frames from {path_file}")
    except Exception as e:
        logging.error(f"Error loading path file: {e}")
        print(f"Error loading path file: {e}.  Make sure you've run path_finder.py first.")
        return

    # --- Get total_frames from user, with validation ---
    print("About to ask for total frames...")
    if num_frames is None:
        total_frames = get_integer_input(f"Enter the total number of frames (max {total_frames_in_path}): ", min_val=1, max_val=total_frames_in_path)
    else:
        total_frames = num_frames
    print(f"Generating {total_frames} frames.") # ADDED PRINT STATEMENT

    # --- Get the absolute path of the current working directory ---
    current_dir = os.path.abspath(".")  # Get absolute path
    logging.info(f"Current working directory: {current_dir}")

    try:
        start_time = time.time()
        frame_num = 0  # Initialize frame_num
        prev_target_x = None  # Initialize previous target coordinates
        prev_target_y = None

        # --- Use a while loop with correct termination condition ---
        while frame_num < total_frames:  # Use user-provided total_frames
            center_x, center_y, zoom = path[frame_num % total_frames_in_path]  # Loop through path if total_frames > path length
            # Convert to float *immediately* after loading from path
            center_x = float(center_x)
            center_y = float(center_y)
            zoom = float(zoom)

            # --- Find an interesting point *around* the current center ---
            target_x, target_y = choose_interesting_point(width, height, center_x, center_y, zoom, max_iter, fractal_type, prev_target_x=prev_target_x, prev_target_y=prev_target_y, julia_c = fractal_math.DEFAULT_JULIA_C if fractal_type == 'julia' else None)

            # --- Smoothly move towards the target point ---
            center_x = (1 - pan_speed) * center_x + pan_speed * target_x
            center_y = (1 - pan_speed) * center_y + pan_speed * target_y

            # --- Use absolute paths for frame filenames ---
            frame_filename = os.path.join(current_dir, f"frame_{frame_num:04d}.png")  # Absolute path
            logging.info(f"Frame: {frame_num}, Zoom: {zoom}, Center: ({center_x}, {center_y}), Filename: {frame_filename}")

            image_renderer.render_fractal_frame_to_png(frame_filename, width, height, center_x, center_y, zoom, max_iter, fractal_type, color_map, noise_scale, noise_strength, noise_octaves, noise_persistence, noise_lacunarity)

            cp._default_memory_pool.free_all_blocks()

            percentage = (frame_num + 1) / total_frames * 100  # Calculate percentage based on user input
            print(f"Frame Progress: {percentage:.2f}%", end="\r")
            elapsed_time = time.time() - start_time
            remaining_time = (elapsed_time / (frame_num + 1)) * (total_frames - frame_num - 1)
            logging.info(f"Frame {frame_num + 1}/{total_frames} rendered. Estimated time remaining: {remaining_time:.2f} seconds")

            time.sleep(render_delay)
            frame_num += 1  # Increment after *successful* frame render
            prev_target_x = target_x  # Update previous target
            prev_target_y = target_y

        print("\nFrame generation complete. Starting FFmpeg...")
        input_pattern = os.path.join(current_dir, 'frame_%04d.png')
        output_file = filename
        ffmpeg_executable = r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"
        fps_str = str(fps)

        time.sleep(1)  # Wait for 1 second to ensure files are written

        ffmpeg_command = [
            ffmpeg_executable,
            '-y',  # Overwrite output file if it exists
            '-framerate', fps_str,
            '-i', input_pattern,
            '-c:v', 'libx264',
            '-crf', '20',
            '-pix_fmt', 'yuv420p',
            output_file
        ]

        try:
            import subprocess
            subprocess.run(ffmpeg_command, check=True, capture_output=True)
            logging.info(f"Video created: {filename}")
            print(f"Video saved as {filename}")

        except subprocess.CalledProcessError as e:
            print("FFmpeg error:")
            print(e.stderr.decode())
            logging.error(f"FFmpeg error: {e.stderr.decode()}")
            raise  # Re-raise the exception

        except FileNotFoundError:
            print(f"Error: FFmpeg executable not found at {ffmpeg_executable}. Please ensure the path is correct.")
            logging.error(f"FFmpeg executable not found at {ffmpeg_executable}")
            raise

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")

    finally:
        # --- Clean up PNG frames (in a finally block) ---
        if num_frames != 1: # Only clean up if it was a video generation
            for i in range(frame_num):  # Iterate up to frame_num (frames actually rendered)
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
    filename = input("Enter the output filename (e.g., my_fractal.mp4 or my_image.png): ")
    path_file = input("Enter the path file name (e.g., zoom_path.npz): ")
    width, height = get_resolution()
    fractal_type = get_fractal_type()
    color_map = get_color_map()
    max_iter = get_integer_input("Enter the maximum number of iterations (higher = more detail, but slower): ", min_val=1, example=20000) # Increased default
    fps = get_integer_input("Enter the frames per second for the video (ignored for single image): ", min_val=1, example=30)
    num_frames = get_integer_input("Enter the total number of frames (1 for single image): ", min_val=1, example=2400)
    render_delay = get_float_input("Enter the render delay in seconds between frames (optional, e.g., 0.05): ", example=0.05)
    pan_speed = get_float_input("Enter the pan speed (0.0 to 1.0, e.g., 0.1): ", min_val=0.0, max_val=1.0, example=0.1)

    # Default values for less commonly changed parameters
    noise_scale = 5.0
    noise_strength = 0.1
    noise_octaves = 6
    noise_persistence = 0.5
    noise_lacunarity = 2.0

    if num_frames == 1:
        base_filename, ext = os.path.splitext(filename)
        if not ext.lower() in ['.png', '.jpg', '.jpeg']:
            filename = base_filename + ".png" # Default to PNG for single image
        generate_single_fractal_image(filename, path_file, width, height, max_iter, fractal_type, color_map, noise_scale, noise_strength, noise_octaves, noise_persistence, noise_lacunarity)
    elif num_frames > 1:
        generate_fractal_video_gpu(filename, path_file, width, height, max_iter, fps, fractal_type, color_map, noise_scale, noise_strength, noise_octaves, noise_persistence, noise_lacunarity, render_delay, pan_speed, num_frames=num_frames)
    else:
        print("Invalid number of frames entered.")