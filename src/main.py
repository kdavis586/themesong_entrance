import numpy as np
import cv2 as cv
import data_collect
import sys
import os

name = input("What is your name? \n")
data_collect.create_dataset(name)
# EXIT_KEY = 27 # ESC

# training_set_path = sys.argv[1]

# # Set up face detection
# face_cascade = cv.CascadeClassifier(training_set_path)

# # Set up webcam feed
# cam = cv.VideoCapture(0, cv.CAP_ANY)
# cam.set(cv.CAP_PROP_FPS, 30)
# cam.set(cv.CAP_PROP_FRAME_WIDTH, 340)
# cam.set(cv.CAP_PROP_FRAME_HEIGHT, 240)

# if not cam.isOpened():
#     raise IOError('Camera could not be opened.')

# frame_count = 0
# # Analyze the footage
# while not cv.waitKey(1) == EXIT_KEY:
#     retval, frame = cam.read()

#     if not retval:
#         raise RuntimeError('Could not retrieve frame')

   
#     # Process frame for detection
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     gray = cv.equalizeHist(gray)

#     faces = face_cascade.detectMultiScale(gray)

#     for (x, y, w, h) in faces:
#         frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
        
#     cv.imshow('Camera Stream', frame)
    
        

# print('Done')
# # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# # gray = cv.equalizeHist(gray)

# # # facial detection uses grayscale

# # # Detect faces in the image
# # faces = face_cascade.detectMultiScale(gray)

# # # Draw rectangles
# # for (x, y, w, h) in faces:
# #     image = cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 5)

# # cv.imshow('Face detecting', image)
# # cv.waitKey(0)


