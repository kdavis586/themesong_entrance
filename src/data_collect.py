import cv2
from datetime import datetime
from PIL import ImageTk, Image
import tkinter as tk
import os

# Defaults for both CamCapture and CamDisplay classes
DEFAULT_RES_WIDTH = 340
DEFAULT_RES_HEIGHT = 240

class CamCapture:
  """Class for live camera feed capture.

  Default values for the following parameters were chosen in mind for 
  Raspberry Pi 4 Model B performance.

  Attributes:
    video_source: An integer for source of video, 0 picks default camera of device
    res_width: An integer for the capture width resolution
    res_height: An integer for the capture height resolution
  """
  def __init__(self, video_source: int = 0, res_width: int = DEFAULT_RES_WIDTH, res_height: int = DEFAULT_RES_HEIGHT):
    """Inits CamCapture with source and capture resolution"""
    self.capture = cv2.VideoCapture(video_source)
    if not self.capture.isOpened():
      raise IOError("Could not open camera: source = ", video_source)

    self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, res_width)
    self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, res_height)

  def __del__(self):
    if self.capture.isOpened():
      self.capture.release()

class CamDisplay:
  """Class for displaying camera feed
  
  Attributes:
    cam_source: A cv2 VideoCapture class to display frames from
    res_width: An integer for the display width resolution
    res_height: An integer for the display height resolution
    display_title: A string for the name of the display window
  """

  def __init__(self, cam_source: cv2.VideoCapture, res_width: int = DEFAULT_RES_WIDTH, res_height: int = DEFAULT_RES_HEIGHT, display_title: str = "Camera Feed"):
    self.cam_source = cam_source
    self.window = tk.Tk()
    self.window.geometry(str(res_width)+"x"+str(res_height)) # Arg must be in "widthxheight" string format
    self.window.title(display_title)

  def open(self):
    self.window.mainloop()
  
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

    #TODO Finish display frames consistently in tkinter window
    



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


