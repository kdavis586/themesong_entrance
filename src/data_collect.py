import cv2
import tkinter
from datetime import datetime
import os

    
def create_dataset(name, path = "src/datasets/"):
    name = __normalize_name(name)
    cam = __init_cam()

    while True:
        retval, frame = cam.read()

        if not retval:
            raise RuntimeError('Could not retrieve frame')

        cv2.imshow('Camera Feed', frame)
        print(cv2.getWindowProperty('Camera Feed', cv2.WND_PROP_VISIBLE))
        key = cv2.waitKey(1)
        # Save image to dataset on SPACE
        if key == 32:
            file_path = __create_filepath(name, path)
            cv2.imwrite(file_path, frame)
            

        # Stop on ESC press or X window button
        if key == 27:
            break

    cam.release()
    cv2.destroyAllWindows()


def __init_cam():
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FPS, 30)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 340)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    if not cam.isOpened():
        raise IOError('Camera could not be opened.')

    return cam

def __create_filepath(name, path):
    file_path = os.path.join(os.getcwd(), path, name + '/')

    # Is there a dataset for this name? Create one if not
    if not os.path.isdir(file_path):
        os.mkdir(file_path)

    file_path += __format_date_name() + name + '.jpg'

    return file_path

def __format_date_name():
    time = datetime.now()
    units = (time.year, time.month, time.day, time.hour, time.minute, time.second)

    # Create string from units of time down to second
    date_str =''
    for unit in units:
        date_str += str(unit)
        date_str += '_'
    
    return date_str

def __normalize_name(name):
    name = ''.join([c for c in name if c.isalpha() or c.isdigit() or c==' ']).rstrip().lower()
    name = name.replace(' ', '_')
    
    return name


