import image_renderer  # Import the new image renderer
import fractal_math
import time
import argparse
import ffmpeg
import logging
import os
import cupy as cp

def generate_fractal_video_gpu(filename, total_frames, width=800, height=600, center_x=-0.5, center_y=0.0, zoom_speed=0.05, max_iter=200, fps=24, fractal_type="mandelbrot", color_map="grayscale", noise_scale=5.0, noise_strength=0.1, noise_octaves=6, noise_persistence=0.5, noise_lacunarity=2.0, render_delay=0.0):
    """Generates a fractal video as a sequence of PNG frames and encodes them into a video."""

    logging.info(f"Generating fractal video: {filename}")
    logging.info(f"Frames: {total_frames}, Width: {width}, Height: {height}, Center: ({center_x}, {center_y}), Zoom Speed: {zoom_speed}, Max Iterations: {max_iter}, FPS: {fps}, Fractal Type: {fractal_type}, Color Map: {color_map}, Noise Scale: {noise_scale}, Noise Strength: {noise_strength}, Noise Octaves: {noise_octaves}, Noise Persistence: {noise_persistence}, Noise Lacunarity: {noise_lacunarity}, Render Delay: {render_delay}")

    try:
        start_time = time.time()
        for frame_num in range(total_frames):
            zoom = 1.0 + frame_num * zoom_speed
            frame_filename = f"frame_{frame_num:04d}.png"  # Use PNG

            image_array_gpu = image_renderer.render_fractal_frame_to_png(frame_filename, width, height, center_x, center_y, zoom, max_iter, fractal_type, color_map, noise_scale, noise_strength, noise_octaves, noise_persistence, noise_lacunarity)

            del image_array_gpu #delete from gpu memory
            cp._default_memory_pool.free_all_blocks()

            percentage = (frame_num + 1) / total_frames * 100
            print(f"Frame Progress: {percentage:.2f}%", end="\r")
            elapsed_time = time.time() - start_time
            remaining_time = (elapsed_time / (frame_num + 1)) * (total_frames - frame_num - 1)
            logging.info(f"Frame {frame_num + 1}/{total_frames} rendered. Estimated time remaining: {remaining_time:.2f} seconds")

            time.sleep(render_delay)  # Delay for cooling

        # Encode directly from the PNG sequence
        (
            ffmpeg
            .input('frame_%04d.png', format='image2', framerate=fps)
            .output(filename, crf=20, pix_fmt='yuv420p')
            .run(quiet=True, overwrite_output=True)
        )
        logging.info(f"Video created: {filename}")

        # Clean up PNG frames
        for frame_num in range(total_frames):
            os.remove(f"frame_{frame_num:04d}.png")
        logging.info("Temporary PNG frames deleted.")

        print(f"\nVideo saved as {filename}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    logging.basicConfig(filename='fractal_generator.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Generate fractal video.")
    parser.add_argument("filename", help="Output video filename.")
    parser.add_argument("total_frames", type=int, help="Total number of frames.")
    parser.add_argument("--width", type=int, default=800, help="Frame width.")
    parser.add_argument("--height", type=int, default=600, help="Frame height.")
    parser.add_argument("--center_x", type=float, default=-0.5, help="Starting center X.")
    parser.add_argument("--center_y", type=float, default=0.0, help="Starting center Y.")
    parser.add_argument("--zoom_speed", type=float, default=0.05, help="Zoom speed.")
    parser.add_argument("--max_iter", type=int, default=200, help="Maximum iterations.")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second of output video.")
    parser.add_argument("--fractal_type", type=str, default="mandelbrot", choices=["mandelbrot", "julia", "burning_ship"], help="Fractal type.")
    parser.add_argument("--color_map", type=str, default="grayscale", choices=["grayscale", "rainbow"], help="Color map.")
    parser.add_argument("--noise_scale", type=float, default=5.0, help="Noise scale (Mandelbrot).")
    parser.add_argument("--noise_strength", type=float, default=0.1, help="Noise strength (Mandelbrot).")
    parser.add_argument("--noise_octaves", type=int, default=6, help="Noise octaves (Mandelbrot).")
    parser.add_argument("--noise_persistence", type=float, default=0.5, help="Noise persistence (Mandelbrot).")
    parser.add_argument("--noise_lacunarity", type=float, default=2.0, help="Noise lacunarity (Mandelbrot).")
    parser.add_argument("--render_delay", type=float, default=0.0, help="Delay between frames (seconds) for cooling.")  # Added render_delay
    args = parser.parse_args()

    generate_fractal_video_gpu(args.filename, args.total_frames, args.width, args.height, args.center_x, args.center_y, args.zoom_speed, args.max_iter, args.fps, args.fractal_type, args.color_map, args.noise_scale, args.noise_strength, args.noise_octaves, args.noise_persistence, args.noise_lacunarity, args.render_delay)