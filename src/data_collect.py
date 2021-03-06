"""Used to create datasets used in facial recognition training.

This module contains functions and classes used to prompt a user
for dataset creation. The user is then showed live feed from their
device's default camera. From there, the user may save photos from the
camera do a dataset folder created for them. This allows for the easy
collection of images of the user to be used in the recognition training.

    Typical usage example:
    
    if user_wants_a_dataset:
        create_dataset(path_to_save_dataset)
"""
import os
import re
from datetime import datetime
from tkinter import Tk, Label
from cv2 import cvtColor, VideoCapture, COLOR_BGR2RGB, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, CAP_ANY
from PIL import ImageTk, Image


# Defaults for both CamCapture and CamDisplay classes
DEFAULT_DATASET_PATH = os.path.join(os.getcwd(), "datasets")
DEFAULT_FRAME_INTERVAL = 10

# Default error message for CamCapture class
CAMERA_SOURCE_ERR_MSG = "Could not open camera."

# Default error message for CamDisplay class
DISPLAY_FRAME_ERR_MSG = "Could not read frame from camera source."

# Prompts for _handle_existing_dataset
EXISTING_DATASET_PROMPT = """Dataset for this name already exists, 
continue to add more samples? (y/n): """
UNKNOWN_RESPONSE_PROMPT = '\nPlease respond with "y" or "n"'

# Prompt for create_dataset
DATASET_PROMPT = "Name to use for new dataset: "


# Module Helper Functions


def _create_dir(name: str, path: str):
    """Creates a directory at the specified path with the given name.

    Args:
        name: A string of the name for the directory
        path: A string representing where the directory will be created

    Returns: A boolean representing if a directory at "path" with "name" has already been created
    """

    already_exists = True
    file_path = os.path.join(path, name)

    # Is there a dataset for this name? Create one if not
    if not os.path.isdir(file_path):
        already_exists = False
        os.mkdir(file_path)

    return already_exists, file_path


def get_time_string():
    """Creates a string representing the current time down to the second.

    Returns: A string representing the current time in the format YYYY_MM_DD_HH_MM_SS
    """

    time = datetime.now()
    units = (time.year, time.month, time.day,
             time.hour, time.minute, time.second)

    # Create string from units of time down to second
    date_str = ""
    for unit in units:
        date_str += str(unit)
        date_str += "_"

    return date_str


def normalize(string: str):
    """Takes a string and normalizes it for future file naming.

    Returns: A string of the normalized input string. Ex. input = "fOO bAr", output = "foo_bar"
    """
    string = string.strip().lower()
    string = re.sub(r"[^(a-z0-9)]", "_", string)

    return string


def _save_frame(name, dataset_path, frame: Image.Image):
    """Saves the ImageTk input frame at dataset path with a standardized name."""
    img_path = os.path.join(dataset_path, f"{get_time_string()}{name}.png")
    frame.save(img_path)


def _handle_existing_dataset():
    """Handles prompting user if they want to extend an existing dataset.

    Returns: A booling representing if the user wants to extend an existing dataset
    """

    while True:
        ans = input(EXISTING_DATASET_PROMPT)
        ans = ans.strip().lower()
        if ans == 'n':
            return False
        elif ans == 'y':
            break
        else:
            print(UNKNOWN_RESPONSE_PROMPT)

    return True

# Module Classes


class CamCapture:
    """Class for live camera feed capture.

    Default values for the following parameters were chosen in mind for
    Raspberry Pi 4 Model B performance.

    Attributes:
                    capture: An integer for source of video, 0 picks default camera of device
                    width: An integer for the capture width resolution
                    height: An integer for the capture height resolution
    """

    def __init__(self):
        """Initializes CamCapture with camera source and capture resolution."""

        # What if the user wants to use a different resolution??
        self.capture = VideoCapture(CAP_ANY)
        self.width = int(self.capture.get(CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(CAP_PROP_FRAME_HEIGHT))

        if not self.capture.isOpened():
            raise IOError(CAMERA_SOURCE_ERR_MSG)

    def close(self):
        """Releases the cv2 VideoCapture object."""
        if self.capture.isOpened():
            self.capture.release()


class CamDisplay:
    """Class for displaying camera feed into a tkinter window.


    Attributes:
        cam_source: The CamCapture object used to get video feed from
        root: The root tkinter window used for display
        video: A tkinter Label class used to show the image within root
    """

    def _centered_tk(self, width_res: int, height_res: int):
        """A helper function that creates a screen-centered
           tkinter window with the given resolution.

        Args:
            win: The tkinter window to center
            width_res: The resolution of the width
            height_res: The resolution of the height
        """

        win = Tk()
        # @TODO Make window resizable later
        win.resizable(width=False, height=False)
        pos_horz = int(win.winfo_screenwidth()/2 - width_res/2)
        pos_vert = int(win.winfo_screenheight()/2 - height_res/2)
        win.geometry(f"{width_res}x{height_res}+{pos_horz}+{pos_vert}")

        return win

    def __init__(self, cam_source: CamCapture = CamCapture(), display_title: str = "Camera Feed"):
        """Initializes CamDisplay with a camera source and window display title.

        Args:
            cam_source: The CamCapture object to used to get video feed from
            display_title: The title used to name the tkinter display window
        """

        self.cam_source = cam_source

        # Set up tk display
        self.root = self._centered_tk(
            self.cam_source.width, self.cam_source.height)
        self.root.title(display_title)
        self.root.bind('<Escape>', lambda event: self.root.quit())
        self.video = Label(self.root)
        self.video.pack()

    def _display_frame(self):
        ret, frame = self.cam_source.capture.read()
        if not ret:
            raise RuntimeError(DISPLAY_FRAME_ERR_MSG)

        # Convert cv2 frame to ImageTk for tkinter window
        frame = cvtColor(frame, COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        # Keep an unaltered frame for save frame
        # (we want to train on the images the camera
        # originally sees, not the flippes ones)
        raw_frame = ImageTk.PhotoImage(frame)

        # Flip display frame across y-axis to display "like a mirror"
        mirror_frame = frame.transpose(Image.FLIP_LEFT_RIGHT)
        mirror_frame = ImageTk.PhotoImage(mirror_frame)

        # Update video label with new frame
        self.video.raw_frame = raw_frame
        self.video.mirror_frame = mirror_frame
        self.video.configure(image=mirror_frame)

        self.root.after(DEFAULT_FRAME_INTERVAL, self._display_frame)

    def show(self):
        """Shows the live feed from cam_source."""
        self._display_frame()
        self.root.mainloop()

# Module Functions


def create_dataset(path: str = DEFAULT_DATASET_PATH):
    """Creates a dataset for a new user / extends a dataset for existing user by allowing
       the user to save images into a dataset folder to be used for classifier training.

       Images are collected via the default camera of the device. Live camera feed is
       shown to the user during dataset collection. The user has the ability to save
       what the camera sees into their dataset folder.

       Controls:
            SPACE: Save current frame of the camera into dataset folder.
            ESC: Quit the dataset application. The same can be achieved by pressing the "X"
                button on the display window GUI.

    Args:
        path: A string representing the path to create the datasets
    """

    name = input(DATASET_PROMPT)
    name = normalize(name)
    # create dataset parent folder if it doesn't already exist
    _create_dir("datasets", os.getcwd())
    # create dataset folder for name if it doesn't already exist
    already_exists, dataset_path = _create_dir(name, path)

    if already_exists:
        # if dataset already exists, see if user wants to extend it
        is_extending = _handle_existing_dataset()
        if not is_extending:
            return

    feed = CamDisplay()
    # bind saving image to spacebar
    feed.root.bind("<space>", lambda event: _save_frame(
        name, dataset_path, ImageTk.getimage(feed.video.raw_frame)))
    feed.show()

    # Release VideoCapture object and destroy opened windows
    feed.cam_source.close()
