import cnn
import data_collect

DEFAULT_WELCOME_MESSAGE = ">>> Roomate Detector <<<\n"
DEFAULT_OPTIONS_MESSAGE = "Type the number of what option you would like to run:\n\n" +\
                          "\t1. Add / Edit a Dataset" +\
                          "\t2. Train the Rommate Detecting Model" +\
                          "\t3. Run Roomate Detector" +\
                          "\t Or type \"exit\" to exit the application"

def run_cli():
    print(DEFAULT_WELCOME_MESSAGE)
    print(DEFAULT_OPTIONS_MESSAGE)
    

    option = input()
    running = True
    
    while running:
        if option == "1":
            data_collect.create_dataset()
        elif option == "2":
            cnn.make_and_train_model()
        elif option == "3":
            # TODO
            print()
        elif option == "exit":
            running = False
        else:
            continue