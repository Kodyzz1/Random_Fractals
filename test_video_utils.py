import unittest
import os
import logging
import video_utils  # Import the new video_utils module

class TestVideoCreator(unittest.TestCase):

    def setUp(self):
        logging.disable(logging.CRITICAL)  # Disable logging during tests

    def tearDown(self):
        logging.disable(logging.NOTSET)  # Re-enable logging

    def test_create_video(self):
        for i in range(3):
            with open(f"test_frame{i:04d}.png", "w") as f:  # Create dummy PNG files
                f.write("dummy image data")  # PNG needs *some* data
        TEST_VIDEO_FILENAME = "test_video.mp4"
        video_utils.create_video("test_frame%04d.png", TEST_VIDEO_FILENAME, framerate=1)
        self.assertTrue(os.path.exists(TEST_VIDEO_FILENAME))
        os.remove(TEST_VIDEO_FILENAME)
        for i in range(3):
            os.remove(f"test_frame{i:04d}.png")

    def test_create_video_overwrite(self):
        TEST_VIDEO_FILENAME = "test_video.mp4"
        with open(TEST_VIDEO_FILENAME, 'w') as f:
            f.write("dummy data")
        video_utils.create_video("test_frame%04d.png", TEST_VIDEO_FILENAME, framerate=1, overwrite=True)
        self.assertTrue(os.path.exists(TEST_VIDEO_FILENAME))
        os.remove(TEST_VIDEO_FILENAME)


    def test_create_video_error(self):
        with self.assertRaises(RuntimeError):
            video_utils.create_video("nonexistent_frame%04d.png", "error_test.mp4")

    def test_create_video_crf_and_pix_fmt(self):
        for i in range(3):
            with open(f"test_frame{i:04d}.png", "w") as f: # Create dummy PNG files
                f.write("dummy image data")  # PNG needs *some* data
        TEST_VIDEO_FILENAME = "test_video_crf.mp4"
        video_utils.create_video("test_frame%04d.png", TEST_VIDEO_FILENAME, framerate=1, crf=18, pix_fmt="yuv444p")
        self.assertTrue(os.path.exists(TEST_VIDEO_FILENAME))
        os.remove(TEST_VIDEO_FILENAME)
        for i in range(3):
            os.remove(f"test_frame{i:04d}.png")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    unittest.main()