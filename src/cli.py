import cnn
import data_collect
import detector
import os

DEFAULT_WELCOME_MESSAGE = ">>> Roomate Detector <<<\n" +\
                          "-------------------------------------------------------------------"
DEFAULT_OPTIONS_MESSAGE = "Type the number of what option you would like to run:\n\n" +\
                          "\t1. Add / Edit a Dataset\n" +\
                          "\t2. Train the Rommate Detecting Model\n" +\
                          "\t3. Run Roomate Detector\n\n" +\
                          "\tOr type \"exit\" to exit the application\n\n"
DEFAULT_INPUT_NOT_RECOGNIZED = "Input not recognized, please try again\n\n"

# TODO Turn this into a class that can potentially have custom paths for data set
def run_cli():
    running = True
    
    while running:
        print(DEFAULT_WELCOME_MESSAGE)
        print(DEFAULT_OPTIONS_MESSAGE)

        option = input()
        option = option.lower()

        if option == "1":
            # What if I dont want to input defaults??? Need to handle that (json?)
            data_collect.create_dataset()
        elif option == "2":
            # TODO how do I get img_width and img_height?
            # Maybe just ask the user (so many vairables in dataset creation (different devices, sources, cameras, etc...))
            cnn.make_and_train_model(data_collect.DEFAULT_DATASET_PATH, os.path.join(os.getcwd(), "model"), "cat_dog", 320, 240)
        elif option == "3":
            detector.detect()
        elif option == "exit":
            running = False
        else:
            print(DEFAULT_INPUT_NOT_RECOGNIZED)