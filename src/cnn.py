"""Used to for all things related to the creation and training of a
Convolution Neural Network (CNN) for image recognition.

This module contains all logic for dataset preprocessing (image preprocessing,
creating training csv), CNN training, and CNN classification.
"""
import csv
import data_collect
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Default tensorflow params
DEFAULT_BATCH_SIZE = 32
DEFAULT_IMG_WIDTH = data_collect.DEFAULT_RES_WIDTH
DEFAULT_IMG_HEIGHT = data_collect.DEFAULT_RES_HEIGHT
DEFAULT_EPOCH_AMOUNT = 15



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

def make_and_train_model(dataset_dir):
    # Creating training and validation datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(DEFAULT_IMG_HEIGHT, DEFAULT_IMG_WIDTH),
        batch_size=DEFAULT_BATCH_SIZE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(DEFAULT_IMG_HEIGHT, DEFAULT_IMG_WIDTH),
        batch_size=DEFAULT_BATCH_SIZE)
    
    class_names = train_ds.class_names
    
    # Configuring Dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Standardizing the dataset
    normalization_layer = layers.Rescaling(1./255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds)) # do I need this???

    num_classes = len(class_names)

    # Use data augmentation to expose model to more samples to reduce overfitting
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                            input_shape=(DEFAULT_IMG_HEIGHT,
                                        DEFAULT_IMG_WIDTH,
                                        3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    # Creating the model
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255, input_shape=(DEFAULT_IMG_HEIGHT, DEFAULT_IMG_WIDTH, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2), # Use dropout to reduce overfitting
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    # Compiling the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Training the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=DEFAULT_EPOCH_AMOUNT
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(DEFAULT_EPOCH_AMOUNT)
    
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    return plt.show()