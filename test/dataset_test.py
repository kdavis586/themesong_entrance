import unittest
import filecmp
import os
import shutil
from tkinter import Tk, Label
from PIL import ImageTk, Image
from datetime import datetime
from src import data_collect


class DataCollectTests(unittest.TestCase):
    """Testing class to test all helper functions, classes, and functions
       in data_collect.py
    """

    def test_create_dir(self):
        """Testing create_dataset creates testing directory"""

        data_collect._create_dir(
            "dir", os.path.join(os.getcwd(), "test"))
        self.assertTrue(os.path.exists("test/dir"))

        # Delete test directory created
        # Only needs to be called if directory creation happened successfully
        os.rmdir(os.path.join(os.getcwd(), "test/dir"))

    def test_get_time_string(self):
        # THIS TEST RELIES ON THE FACT THAT IT WILL RUN IN UNDER 1 SECOND
        # IF IT DOES NOT, IT WILL FAIL
        time = datetime.now()
        units = (time.year, time.month, time.day,
                 time.hour, time.minute, time.second)
        expected = ""
        for unit in units:
            expected += str(unit)
            expected += "_"

        actual = data_collect._get_time_string()
        self.assertEqual(expected, actual)

    def test_normalize_string_lowercase(self):
        expected = "test"
        actual = data_collect._normalize("test")

        self.assertEqual(expected, actual)

    def test_normalize_string_uppercase(self):
        expected = "test"
        actual = data_collect._normalize("TEST")

        self.assertEqual(expected, actual)

    def test_normalize_string_space_replace(self):
        expected = "t_e_s_t"
        actual = data_collect._normalize("t e s t")

        self.assertEqual(expected, actual)

    def test_normalize_string_forbidden_ascii(self):
        expected = "__________"  # Length = 10
        actual = data_collect._normalize('<>:"/\\|\?*')

        self.assertEqual(expected, actual)

    def test_normalize_string_control_chars(self):
        test_string = ""
        expected = "_________________________________"  # Length = 33
        control_codes = [chr(i) for i in range(32)]
        control_codes.append(chr(127))

        for code in control_codes:
            test_string += code

        actual = data_collect._normalize(test_string)

        self.assertEqual(expected, actual)

    def test_normalize_string_extended_ascii(self):
        expected = "_____"  # Length = 5
        actual = data_collect._normalize('ĊæäÀɐ')

        self.assertEqual(expected, actual)

    def test_normalize_string_emoji(self):
        expected = "_____"  # Length = 5
        actual = data_collect._normalize("😂😎😑🤗🤐")

        self.assertEqual(expected, actual)

    def test_normalize_string_emoticon(self):
        expected = "____"  # Length = 4
        actual = data_collect._normalize("⊙﹏⊙∥")

        self.assertEqual(expected, actual)

    def test_save_frame_saves_img(self):
        test_dir_path = "test/test_dir"
        test_img_path = "test/test_images/test_img_1.png"
        test_img = Image.open(test_img_path)
        save_path = os.path.join(os.getcwd(), test_dir_path)
        os.mkdir(save_path)
        data_collect._save_frame("test", save_path, test_img)

        # Check to see if save directory has a file in it
        is_empty = not bool(os.listdir(save_path))
        shutil.rmtree(test_dir_path)

        if is_empty:
            raise AssertionError(
                "Save directory did not contain a saved image.")

    # @TODO: create tests for _handle_existing_dataset (bool outputs, and cli outputs in response to bad input)


if __name__ == '__main__':
    unittest.main()
