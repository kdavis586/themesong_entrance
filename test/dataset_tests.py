import unittest
from os import path
from src.data_collect import *


class DatasetTests(unittest.TestCase):
    # Testing create_dataset creates testing directory
    def test_path_creation(self):
        create_dataset("testing", "test/test_dir")
        self.assertTrue(path.exists("test/test_dir"))

if __name__ == '__main__':
    unittest.main()