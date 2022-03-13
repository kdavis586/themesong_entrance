"""Used to for all things related to the creation and training of a
Convolution Neural Network (CNN) for image recognition.

This module contains all logic for dataset preprocessing (image preprocessing,
creating training csv), CNN training, and CNN classification.
"""
import pandas as pd
import numpy as np
import csv
import os
import tensorflow as tf


def _create_cnn_csv(path: str, csv_name: str = "labeled_data"):
    """Creates one csv with paths to all images in all datasets with the correct corresponding labels.
    This function is to be used to have a dataset for training/validation.

    Note: This function will rely on the fact that your datasets folder follows the following directory
    layout:

    parent_folder
        | - label_for_example_1
        |   | - image_for_example_1
        |   | - another_image_for_example_1

        | - label_for_example_2
        |   | - image_for_example_2
        |   | - etc...

    As you can see, the folder name that contains images of a particular person/object will be used as the label.

    Args:
        path: The path to parent folder which contains all of the datasets
        csv_name: The name for the csv file that will be created (without .csv in the name)
    """
    csv_path = os.path.join(path, f"{csv_name}.csv")

    with open(csv_path, "w") as labeled_data:
        writer = csv.writer(labeled_data, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for dirname in os.listdir(path):
            dirpath = os.path.join(path, dirname)

            if os.path.isdir(dirpath):
                # Loop through all files in this folder
                for filename in os.listdir(dirpath):
                    filepath = os.path.join(dirpath, filename)

                    if os.path.isfile(filepath):
                        # dirname will serve as the label for all files in it
                        writer.writerow([filepath, dirname])

def something():
    print("This is garbage...")