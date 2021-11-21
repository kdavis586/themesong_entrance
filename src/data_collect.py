import cv2
from datetime import datetime
from PIL import ImageTk, Image
import Tkinter as tk
import os

# Defaults for both CamCapture and CamDisplay classes
DEFAULT_RES_WIDTH = 340
DEFAULT_RES_HEIGHT = 240
DEFAULT_IMG_SAVE_PATH = "src/datasets/"

class CamCapture:
  """Class for live camera feed capture.

  Default values for the following parameters were chosen in mind for 
  Raspberry Pi 4 Model B performance.

  Attributes:
    video_source: An integer for source of video, 0 picks default camera of device
    width: An integer for the capture width resolution
    height: An integer for the capture height resolution
  """
  def __init__(self, video_source: int = 0, width: int = DEFAULT_RES_WIDTH, height: int = DEFAULT_RES_HEIGHT):
    """Inits CamCapture with source and capture resolution"""
    self.capture = cv2.VideoCapture(video_source)
    self.width = width
    self.height = height

    if not self.capture.isOpened():
      raise IOError("Could not open camera: source = ", video_source)

    self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
    self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

  def __del__(self):
    if self.capture.isOpened():
      self.capture.release()

class CamDisplay:
  """Class for displaying camera feed
  
  Attributes:
    cam_source: A CamCapture object to get video feed from
    display_title: A string for the name of the display window
  """

  def __init__(self, cam_source: CamCapture = CamCapture(), display_title: str = "Camera Feed"):
    self.cam_source = cam_source
    self.window = tk.Tk()
    self.window.geometry(f"{cam_source.width}x{cam_source.height}") # Arg must be in "widthxheight" string format
    self.window.title(display_title)

  def open(self):
    self.window.mainloop()
  
  def close():
    self.window.destroy()
  
  def display_frame(self):
    ret, frame = self.cam_source.read()
    if not ret:
      raise RuntimeError("Could not read frame from camera source")

    # Convert to RGB for PIL
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to PIL
    frame = Image.fromarray(frame)

    # Convert to ImageTk
    frame = ImageTk.PhotoImage(frame)

    # Create label for image display
    panel = tk.Label(self.window, image=frame)
    panel.pack(side = "bottom", fill="both", expand="yes")

    # Return frame displayed
    return frame
    



def create_dataset(name, path = DEFAULT_IMG_SAVE_PATH):
    name = __normalize_name(name)
    display = CamDisplay()
    display.open()

    while True:
        frame = self.cam_source.display_frame()
        # Get the last byte of waitkey to avoid modifier key interferance
        key = cv2.waitKey(1) & 0xFF
        # Save image to dataset on SPACE
        if key == 32:
            file_path = __create_filepath(name, path)
            cv2.imwrite(file_path, frame)
            

        # Stop on ESC press or X window button
        if key == 27:
            break

    display.close()


def __create_filepath(name, path):
    file_path = os.path.join(os.getcwd(), path, f"{name}/")

    # Is there a dataset for this name? Create one if not
    if not os.path.isdir(file_path):
        os.mkdir(file_path)

    file_path = f"{file_path}{__format_date_name()}{name}.jpg"

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


