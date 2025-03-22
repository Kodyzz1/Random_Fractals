import ffmpeg
import os
import unittest
import logging

def create_video(input_pattern, output_filename, framerate=30, overwrite=True, crf=20, pix_fmt='yuv420p'):
    """
    Creates a video from a sequence of SVG frames using ffmpeg-python.

    Args:
        input_pattern (str): Input file pattern (e.g., 'frame%04d.svg').
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
    logging.info(f"Video creation completed: {output_filename}")

class TestVideoCreator(unittest.TestCase):

    def setUp(self):
        logging.disable(logging.CRITICAL) # disable logging during test

    def tearDown(self):
        logging.disable(logging.NOTSET) # re-enable logging.

    def test_create_video(self):
        for i in range(3):
            with open(f"test_frame{i:04d}.svg", "w") as f:
                f.write("<svg></svg>")
        TEST_VIDEO_FILENAME = "test_video.mp4"
        create_video("test_frame%04d.svg", TEST_VIDEO_FILENAME, framerate=1)
        self.assertTrue(os.path.exists(TEST_VIDEO_FILENAME))
        os.remove(TEST_VIDEO_FILENAME)
        for i in range(3):
            os.remove(f"test_frame{i:04d}.svg")

    def test_create_video_overwrite(self):
        TEST_VIDEO_FILENAME = "test_video.mp4"
        with open(TEST_VIDEO_FILENAME, 'w') as f:
            f.write("dummy data")
        create_video("test_frame%04d.svg", TEST_VIDEO_FILENAME, framerate=1, overwrite=True)
        self.assertTrue(os.path.exists(TEST_VIDEO_FILENAME))
        os.remove(TEST_VIDEO_FILENAME)

    def test_create_video_error(self):
        with self.assertRaises(RuntimeError):
            create_video("nonexistent_frame%04d.svg", "error_test.mp4")

    def test_create_video_crf_and_pix_fmt(self):
        for i in range(3):
            with open(f"test_frame{i:04d}.svg", "w") as f:
                f.write("<svg></svg>")
        TEST_VIDEO_FILENAME = "test_video_crf.mp4"
        create_video("test_frame%04d.svg", TEST_VIDEO_FILENAME, framerate=1, crf=18, pix_fmt="yuv444p")
        self.assertTrue(os.path.exists(TEST_VIDEO_FILENAME))
        os.remove(TEST_VIDEO_FILENAME)
        for i in range(3):
            os.remove(f"test_frame{i:04d}.svg")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    unittest.main()