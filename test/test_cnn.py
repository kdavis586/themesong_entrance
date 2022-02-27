import unittest
import os
import csv
from src import cnn

TEST_CSV_1_VALUES = [[os.path.join(os.getcwd(), "test", "test_images", "test_labeled_folder", "test_img_1.png"), "test_labeled_folder"]]


class CnnTests(unittest.TestCase):
    """Testing class to test all helper functions, classes, and functions
       in cnn.py
    """

    def test_create_cnn_csv_1(self):
        test_path = os.path.join(os.getcwd(), "test", "test_images")
        # name for the actual output csv
        actual_label = "test_labeled_data"
        cnn._create_cnn_csv(test_path, actual_label)

        expected_csv_path = os.path.join(test_path, f"{actual_label}.csv")

        self.assertTrue(os.path.exists(expected_csv_path))
        self.assertTrue(os.path.isfile(expected_csv_path))

        # Getting contents of csv, then deleting in case of failure below
        actual_csv_values = []
        with open(expected_csv_path, 'rt') as actual_csv:
            reader = csv.reader(actual_csv)

            for row in reader:
                # ignoring empty rows
                if row == []:
                    continue 
                actual_csv_values.append(row)

        os.remove(expected_csv_path)

        # Comparing the contents of expected and actual csvs
        self.assertTrue(len(actual_csv_values) == len(TEST_CSV_1_VALUES), f"{len(actual_csv_values)} is not {len(TEST_CSV_1_VALUES)}")

        for i in range(len(actual_csv_values)):
            self.assertTrue(TEST_CSV_1_VALUES[i] == actual_csv_values[i], f"{actual_csv_values[i]} is not {TEST_CSV_1_VALUES[i]}")