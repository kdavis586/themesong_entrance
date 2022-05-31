"""


 TODO FIX ALL OF THIS FUCKING GARBAGE, SUPER REPEATED CODE.


"""

import numpy as np
import os
import tensorflow as tf

from cv2 import cvtColor, COLOR_BGR2RGB
from data_collect import CamCapture, DEFAULT_FRAME_INTERVAL, DISPLAY_FRAME_ERR_MSG
from PIL import ImageTk, Image
from tkinter import Tk, Label



class DetectCamDisplay:
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

    def __init__(self, model_path, cam_source: CamCapture = CamCapture(), display_title: str = "Camera Feed"):
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

        # Tensorflow
        self.model = tf.keras.models.load_model(model_path)
        self.classnames = ["cat", "dog"]

    def _display_frame(self):
        ret, frame = self.cam_source.capture.read()
        if not ret:
            raise RuntimeError(DISPLAY_FRAME_ERR_MSG)

        # Convert cv2 frame to Image
        frame = cvtColor(frame, COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        print(f'frame size: {frame.size}')
        self.video.frame_image = frame

        # Flip display frame across y-axis to display "like a mirror", covert to ImageTk for display
        mirror_frame = frame.transpose(Image.FLIP_LEFT_RIGHT)
        mirror_frame = ImageTk.PhotoImage(mirror_frame)

        # Update video label with new frame
        self.video.mirror_frame = mirror_frame
        self.video.configure(image=mirror_frame)

        self.root.after(DEFAULT_FRAME_INTERVAL, self._display_frame)

    def show(self):
        """Shows the live feed from cam_source."""
        self._display_frame()

        img_arr = tf.keras.preprocessing.image.img_to_array(self.video.frame_image)
        img_arr = tf.expand_dims(img_arr, 0)
        predictions = self.model.predict(img_arr)
        score = tf.nn.softmax(predictions[0])
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(self.class_names[np.argmax(score)], 100 * np.max(score))
        )

        self.root.mainloop()
    
def detect():
    model_path = os.path.join(os.getcwd(), "model", "cat_dog_model.h5")
    display = DetectCamDisplay(model_path)
    display.show()

# (None, 240, 340, 3), found shape=(None, 240, 320, 3)