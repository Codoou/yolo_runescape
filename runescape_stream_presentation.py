import cv2
import numpy as np
import time

from mss import mss
from PIL import Image
from pynput.mouse import Button, Controller
from pynput.keyboard import Controller as Kontroller
from ultralytics import YOLO

from runescape_helper_function import *

# full runelite client
CLIENT_COORDINATES = {"top": 70, "left": 5, "width": 765, "height": 505}

# frames per second
FPS = .01
# Weight File
WEIGHTS = r'C:\Projects\image-detection\runs\detect\train8\weights\best.pt'

# model
model = YOLO(WEIGHTS)

mouse = Controller()
keyboard = Kontroller()

# setting up screen shotting
with mss() as sct:
    frame_number = 0
    while True:
        
        # for calculating ftp
        start_time = time.time()
        print("================================================")

        # Get screen shot and convert to numpy array
        img = np.asarray(sct.grab(CLIENT_COORDINATES))
        
        # fix colors and convert to 3 channel vs 4 channel (remove A)
        image_copy = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        gray_copy = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # run prediction against image
        results = model(image_copy)

        # Plot bounding boxes to image
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        # Calculation for FPS
        elapsed_time = time.time() - start_time

        if elapsed_time < FPS:
            time.sleep(FPS - elapsed_time)
        

        frame_number += 1