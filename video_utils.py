import ffmpeg
import logging

def create_video(input_pattern: str, output_filename: str, framerate: int = 30, overwrite: bool = True, crf: int = 20, pix_fmt: str = 'yuv420p'):
    """
    Creates a video from a sequence of image files using ffmpeg-python.

    Args:
        input_pattern (str): Input file pattern (e.g., 'frame%04d.png').
        output_filename (str): Output video filename (e.g., 'output.mp4').
        framerate (int): Frames per second.
        overwrite (bool): Whether to overwrite an existing output file.
        crf (int): Constant Rate Factor (0-51, lower is better quality).
        pix_fmt (str): Pixel format.
    """
    logging.info(f"Starting video creation: {output_filename}, FPS: {framerate}, CRF: {crf}, PixFmt: {pix_fmt}")
    try:
        (
            ffmpeg
            .input(input_pattern, framerate=framerate)
            .output(output_filename, overwrite_output=overwrite, crf=crf, pix_fmt=pix_fmt)
            .run(quiet=True)
        )
        logging.info(f"Video created: {output_filename}")
    except ffmpeg.Error as e:
        logging.error(f"ffmpeg error: {e.stderr.decode()}")
        raise RuntimeError(f"ffmpeg error: {e.stderr.decode()}")