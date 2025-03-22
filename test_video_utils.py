import unittest
import os
import logging
import video_utils
import imageio.v3 as iio
import numpy as np

class TestVideoCreator(unittest.TestCase):

    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.test_dir = "test_temp_dir"  # Use a temporary directory
        os.makedirs(self.test_dir, exist_ok=True)


    def tearDown(self):
        logging.disable(logging.NOTSET)
        # Clean up the temporary directory and its contents
        if os.path.exists(self.test_dir):
            for filename in os.listdir(self.test_dir):
                file_path = os.path.join(self.test_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path) #for recursive deletion
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
            os.rmdir(self.test_dir)



    def test_create_video(self):
        # Create valid dummy PNG files within the test directory
        for i in range(3):
            dummy_image = np.zeros((10, 10, 3), dtype=np.uint8)
            iio.imwrite(os.path.join(self.test_dir, f"test_frame{i:04d}.png"), dummy_image)

        TEST_VIDEO_FILENAME = os.path.join(self.test_dir, "test_video.mp4")
        video_utils.create_video(os.path.join(self.test_dir, "test_frame%04d.png"), TEST_VIDEO_FILENAME, framerate=1)
        self.assertTrue(os.path.exists(TEST_VIDEO_FILENAME))



    def test_create_video_overwrite(self):
        TEST_VIDEO_FILENAME = os.path.join(self.test_dir, "test_video.mp4")
        with open(TEST_VIDEO_FILENAME, 'w') as f:
            f.write("dummy data")

         # Create valid dummy PNG files within the test directory
        for i in range(3):
            dummy_image = np.zeros((10, 10, 3), dtype=np.uint8)
            iio.imwrite(os.path.join(self.test_dir, f"test_frame{i:04d}.png"), dummy_image)

        video_utils.create_video(os.path.join(self.test_dir, "test_frame%04d.png"), TEST_VIDEO_FILENAME, framerate=1, overwrite=True)
        self.assertTrue(os.path.exists(TEST_VIDEO_FILENAME))




    def test_create_video_error(self):
        with self.assertRaises(RuntimeError):
            video_utils.create_video(os.path.join(self.test_dir, "nonexistent_frame%04d.png"), os.path.join(self.test_dir, "error_test.mp4"))

    def test_create_video_crf_and_pix_fmt(self):
        # Create valid dummy PNG files within the test directory
        for i in range(3):
            dummy_image = np.zeros((10, 10, 3), dtype=np.uint8)
            iio.imwrite(os.path.join(self.test_dir, f"test_frame{i:04d}.png"), dummy_image)

        TEST_VIDEO_FILENAME = os.path.join(self.test_dir, "test_video_crf.mp4")
        video_utils.create_video(os.path.join(self.test_dir, "test_frame%04d.png"), TEST_VIDEO_FILENAME, framerate=1, crf=18, pix_fmt="yuv444p")
        self.assertTrue(os.path.exists(TEST_VIDEO_FILENAME))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    unittest.main()