import cv2
from datetime import datetime
from PIL import ImageTk, Image
import tkinter as tk
import os

# Defaults for both CamCapture and CamDisplay classes
DEFAULT_RES_WIDTH = 340
DEFAULT_RES_HEIGHT = 240
DEFAULT_DATASET_PATH = os.path.join(os.getcwd(), "datasets/")
DEFAULT_FRAME_INTERVAL = 10

# Module Helper Functions


def _create_filepath(name, path):
    already_exists = True
    file_path = os.path.join(path, f"{name}/")

    # Is there a dataset for this name? Create one if not
    if not os.path.isdir(file_path):
        already_exists = False
        os.mkdir(file_path)

    return already_exists, file_path


def _get_time_string():
    time = datetime.now()
    units = (time.year, time.month, time.day,
             time.hour, time.minute, time.second)

    # Create string from units of time down to second
    date_str = ''
    for unit in units:
        date_str += str(unit)
        date_str += '_'

    return date_str


def _normalize_name(name):
    name = ''.join([c for c in name if c.isalpha()
                    or c.isdigit() or c == ' ']).strip().lower()
    name = name.replace(' ', '_')

    return name


def _save_frame(name, dataset_path, frame: ImageTk.PhotoImage):
    img = ImageTk.getimage(frame)
    img_path = f"{dataset_path}{_get_time_string()}{name}.png"
    img.save(img_path)

    # cv2.imwrite(img_path, frame)


"""Handles prompting user if they want to extend an existing dataset"""


def _existing_dataset_prompt():
    while True:
        ans = input(
            "Dataset for this name already exists, continue to add more samples? (y/n): ")
        ans = ans.strip().lower()
        if ans == 'n':
            return False
        elif ans == 'y':
            break
        else:
            print('\nPlease respond with "y" or "n"')

    return True

# Module Classes


class CamCapture:
    """Class for live camera feed capture.

    Default values for the following parameters were chosen in mind for
    Raspberry Pi 4 Model B performance.

    Attributes:
                    video_source: An integer for source of video, 0 picks default camera of device
                    width: An integer for the capture width resolution
                    height: An integer for the capture height resolution
    """

    def __init__(self, video_source: int = 0, width: int = DEFAULT_RES_WIDTH,
                 height: int = DEFAULT_RES_HEIGHT):
        """Inits CamCapture with source and capture resolution"""
        self.capture = cv2.VideoCapture(0, video_source)
        self.width = width
        self.height = height

        if not self.capture.isOpened():
            raise IOError(f"Could not open camera: source = {video_source}")

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def close(self):
        if self.capture.isOpened():
            self.capture.release()


class CamDisplay:

    # Class helper functions
    """Creates a centered tkinter window at specified resolution

                Args:
            win: The tkinter window to center
            width_res: The resolution of the width
            height_res: The resolution of the height
        """

    def _centered_tk(self, width_res: int, height_res: int):
        win = tk.Tk()
        # @TODO Make window resizable later
        win.resizable(width=False, height=False)
        pos_horz = int(win.winfo_screenwidth()/2 - width_res/2)
        pos_vert = int(win.winfo_screenheight()/2 - height_res/2)
        win.geometry(f"{width_res}x{height_res}+{pos_horz}+{pos_vert}")

        return win

    # Class functions
    """Class for displaying camera feed

    Args:
                    cam_source: A CamCapture object to get video feed from
                    display_title: A string for the name of the display window
    """

    def __init__(self, cam_source: CamCapture = CamCapture(), display_title: str = "Camera Feed"):
        self.cam_source = cam_source

        # Set up tk display
        self.root = self._centered_tk(
            self.cam_source.width, self.cam_source.height)
        self.root.title(display_title)
        self.root.bind('<Escape>', lambda event: self.root.quit())
        self.video = tk.Label(self.root)
        self.video.pack()

    def display_frame(self):
        ret, frame = self.cam_source.capture.read()
        if not ret:
            raise RuntimeError("Could not read frame from camera source")

        # Convert cv2 frame to ImageTk for tkinter window
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(frame)

        # Update video label with new frame
        self.video.image = frame
        self.video.configure(image=frame)

        self.root.after(DEFAULT_FRAME_INTERVAL, self.display_frame)

    def show(self):
        self.display_frame()
        self.root.mainloop()

# Module Functions


def create_dataset(path=DEFAULT_DATASET_PATH):
    name = input("Name to use for new dataset: ")
    name = _normalize_name(name)
    # create dataset folder for name if it doesn't already exist
    already_exists, dataset_path = _create_filepath(name, path)
    if already_exists and not _existing_dataset_prompt():
        return

    feed = CamDisplay()
    # bind saving image to spacebar
    feed.root.bind("<space>", lambda event: _save_frame(
        name, dataset_path, feed.video.image))
    feed.show()

    # Release VideoCapture obj and destroy opened windows
    feed.cam_source.close()
