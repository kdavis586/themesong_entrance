"""Used to for all things related to the creation and training of a
Convolution Neural Network (CNN) for image recognition.

This module contains all logic for dataset preprocessing (image preprocessing,
creating training csv), CNN training, and CNN classification.
"""
from statistics import mode
import data_collect
import matplotlib.pyplot as plt
import os
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Default tensorflow params
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCH_AMOUNT = 15

# CLI save handling strings
DEFAULT_MODEL_EXITS_MESSAGE_ARR = ["A trained model with the name ",
                                   "A result graph with the name ",
                                   " already exists, enter the number of the option you would like to use:\n" +\
                                   "\t1. Overwrite the existing file.\n" +\
                                   "\t2. Save the new file with a temporary generated name.\n"]


def make_and_train_model(dataset_dir, model_output_dir, model_name, img_width, img_height):
    """ Creates and traines a tf.keras.Sequential model and saves it 
    so it can be loaded and reused (in HD5F format).

    Args:
        dataset_dir: The path of the dataset
                     directory (containing all subfolders with training examples)
        model_output_dir: The path to save the model and model analytics.
        model_name: The name of the model when saved into model_output_dir
    """
    # Handle if either input path does not exist
    if not os.path.exists(dataset_dir):
        print(f"dataset_dir: \"{dataset_dir}\" either could not be found or does not exist!")
        return

    if not os.path.exists(model_output_dir):
        print(f"model_output_dir: \"{model_output_dir}\" either could not be found or does not exist!")
        return

    # Creating training and validation datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=DEFAULT_BATCH_SIZE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=DEFAULT_BATCH_SIZE)
    
    class_names = train_ds.class_names
    
    # Configuring Dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = len(class_names)

    # Use data augmentation to expose model to more samples to reduce overfitting
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                            input_shape=(img_height,
                                        img_width,
                                        3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    # Creating the model (includes data standardization via layers.Rescaling)
    # Also includes Drop to reduce overfitting
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
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

    # Creating Training Results Plot
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(DEFAULT_EPOCH_AMOUNT)
    
    result_graph = plt.figure(figsize=(8, 8))
    result_graph.add_subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    _handle_save_data(model, result_graph, model_output_dir, model_name)

def _handle_save_data(trained_model, result_graph, model_output_dir, model_name):
    """ Handles situations of saving model that might occur during runtime.
    Gives options on how to proceed to the user in the event we are trying to save to 
    the same file location of another model.

    Options for the user include:
        1. Overwriting the current model in the conflicting path.
        2. Choosing a new model name to avoid the conflict.
        3. Discarding trained_model.
    
    Args:
        trained_model: The model that is going to be potentially saved.
        result_graph: The plot that shows training results.
        model_output_dir: The directory in which to save the model.
        model_name: The name to be used for saving trained_model and result_graph.
    """
    model_name = data_collect.normalize(model_name)
    model_path = os.path.join(model_output_dir, f"{model_name}_model.h5")
    graph_path = os.path.join(model_output_dir, f"{model_name}_graph.jpg")

    model_path_exists = os.path.exists(model_path)
    graph_path_exists = os.path.exists(graph_path)

    if model_path_exists:
        model_path = _handle_model_graph_path_exists(model_output_dir, model_path, model_name, True)
    
    if graph_path_exists:
        graph_path = _handle_model_graph_path_exists(model_output_dir, graph_path, model_name, False)
    
    trained_model.save(model_path)
    result_graph.savefig(graph_path)

def _handle_model_graph_path_exists(output_dir, filepath, name, is_model):
    """ Does the cli handling in the event there is a model/graph at filepath.
    
    Args:
        output_dir: The path to the directory to save the model
        filepath: The path that has a conflicting model file
        name: The current name of the model to be used for saving
        is_graph: Whether or not the file conflict is for the model (if False, conflict is for the graph)
    
    Returns:
        A path that has no file conflicts that is safe for saving.
    """
    filepath_exists = True

    while filepath_exists:
        if is_model:
            message = ''.join([DEFAULT_MODEL_EXITS_MESSAGE_ARR[0], f"\"{name}\"", DEFAULT_MODEL_EXITS_MESSAGE_ARR[2]])
        else:
            message = ''.join([DEFAULT_MODEL_EXITS_MESSAGE_ARR[1], f"\"{name}\"", DEFAULT_MODEL_EXITS_MESSAGE_ARR[2]])
        print(message)

        option = input()

        if option == "1":
            # Overwrite the model data currently at model_path
            os.remove(filepath)
            filepath_exists = os.path.exists(filepath)  
        elif option == "2":
            # Generate a temporary path for the model to save under
            name = data_collect.get_time_string()
            if is_model:
                filepath = os.path.join(output_dir, f"{name}_model.h5")
            else:
                filepath = os.path.join(output_dir, f"{name}_graph.jpg")
            filepath_exists = os.path.exists(filepath)  
        else:
            # Input not understood
            print("Option not recognized, please type in the number of the option you want.\n\n")
    
    return filepath