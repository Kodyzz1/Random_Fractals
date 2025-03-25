import numpy as np
import matplotlib.pyplot as plt
import random
from fractal_math import mandelbrot  # Import the CPU version

def mandelbrot(c, max_iter):
    # (Your existing mandelbrot function - you might want to rename this to something like mandelbrot_gpu if you still have it)
    z = 0.0j
    iterations = np.zeros(c.shape, dtype=int)
    for i in range(max_iter):
        z = z*z + c
        mask = (np.abs(z) < 2) & (iterations == 0)
        iterations[mask] = i + 1
        z[~mask] = 2
    return iterations

def choose_interesting_point_for_path(center_x, center_y, zoom, max_iter, fractal_type='mandelbrot'):
    """
    Chooses an interesting point near the current center for path generation.
    Uses a low-resolution CPU Mandelbrot calculation.
    """
    search_radius_factor = 0.5
    search_area_zoom = zoom * search_radius_factor
    search_width = 100
    search_height = 100

    x_coords = np.linspace(center_x - search_area_zoom * (search_width / search_height), center_x + search_area_zoom * (search_width / search_height), search_width)
    y_coords = np.linspace(center_y - search_area_zoom, center_y + search_area_zoom, search_height)
    xv, yv = np.meshgrid(x_coords, y_coords)
    c = xv + 1j * yv

    if fractal_type == 'mandelbrot':
        iterations = mandelbrot(c, max_iter)
    else:
        return None, None # Only Mandelbrot supported for now in path finder

    boundary_points_indices = np.where((iterations > 0) & (iterations < max_iter))
    if boundary_points_indices[0].size > 0:
        index = random.choice(range(boundary_points_indices[0].size))
        target_x = x_coords[boundary_points_indices[1][index]]
        target_y = y_coords[boundary_points_indices[0][index]]
        return target_x, target_y
    return None, None

def find_path(start_x, start_y, initial_zoom, zoom_factor, num_frames, max_iter, filename="zoom_path.npz", pan_speed=0.02, target_update_interval=50):
    """Finds a path by periodically panning towards an interesting point and zooming."""

    path = []
    zoom = initial_zoom
    center_x = start_x
    center_y = start_y
    current_target_x = None
    current_target_y = None

    for i in range(num_frames):
        path.append((center_x, center_y, zoom))

        # Update the target point every target_update_interval frames
        if i % target_update_interval == 0:
            new_target_x, new_target_y = choose_interesting_point_for_path(center_x, center_y, zoom, max_iter)
            if new_target_x is not None and new_target_y is not None:
                current_target_x = new_target_x
                current_target_y = new_target_y
                print(f"New target found at frame {i}: ({current_target_x:.6f}, {current_target_y:.6f})")
            else:
                print(f"Warning: Could not find an interesting point at frame {i}. Continuing with the current target.")

        # Move towards the current target point if one exists
        if current_target_x is not None and current_target_y is not None:
            center_x = (1 - pan_speed) * center_x + pan_speed * current_target_x
            center_y = (1 - pan_speed) * center_y + pan_speed * current_target_y

        zoom *= zoom_factor

    np.savez(filename, path=np.array(path))
    print(f"Path saved to {filename}")
    return path

def visualize_path(path, max_iter, filename="path_visualization.png"):
    # (Your existing visualize_path function - no changes needed for this example)
    if not path:
        print("Error: Path is empty, can't visualize.")
        return

    width = 1000
    height = 1000
    center_x = -0.5
    center_y = 0.0
    zoom = 2.0

    x = np.linspace(center_x - zoom, center_x + zoom, width)
    y = np.linspace(center_y - zoom, center_y + zoom, height)
    xv, yv = np.meshgrid(x, y)
    c = xv + 1j * yv
    mandelbrot_image = mandelbrot(c, max_iter)

    plt.figure(figsize=(10, 10))
    plt.imshow(mandelbrot_image, extent=[x.min(), x.max(), y.min(), y.max()], cmap='magma', origin='lower')

    path_xs, path_ys, path_zooms = zip(*path)
    path_zooms = np.array(path_zooms)
    norm_zooms = np.log(path_zooms / np.min(path_zooms))
    norm_zooms = norm_zooms / np.max(norm_zooms)
    sc = plt.scatter(path_xs, path_ys, c=norm_zooms, cmap='viridis', s=20, edgecolors='w', linewidths=0.5)
    cbar = plt.colorbar(sc, label='Zoom Level')
    tick_locs = np.linspace(0, 1, 5)
    tick_labels = [f"{np.exp(t * np.log(path_zooms.max() / path_zooms.min())) * path_zooms.min():.2f}" for t in tick_locs]
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(tick_labels)
    plt.title("Zoom Path Visualization (Color-coded by Zoom)")
    plt.xlabel("Re(c)")
    plt.ylabel("Im(c)")
    plt.savefig(filename)
    plt.close()
    print(f"Path visualization saved to {filename}")

if __name__ == '__main__':
    start_x = -0.745
    start_y = 0.112
    initial_zoom = 0.005
    zoom_factor = 0.97
    max_iter = 200  # Lower for faster path finding
    path_file = "zoom_path.npz"
    pan_speed = 0.01 # Adjust for panning speed in path generation
    target_update_interval = 50 # Update the target every 50 frames

    num_frames_input = input("Enter the number of frames for the path (e.g., 500): ")
    try:
        num_frames = int(num_frames_input)
    except ValueError:
        print("Invalid input. Using default of 500 frames.")
        num_frames = 500

    path = find_path(start_x, start_y, initial_zoom, zoom_factor, num_frames, max_iter, path_file, pan_speed, target_update_interval)
    visualize_path(path, max_iter)