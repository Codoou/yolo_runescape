import cv2
import numpy as np
import time

from mss import mss
from PIL import Image
from pynput.mouse import Button, Controller
from pynput.keyboard import Controller as Kontroller
from ultralytics import YOLO

from runescape_helper_function import *

# only the playable screen
#mon = {"top": 70, "left": 5, "width": 520, "height": 340}

STATIC_IMAGE_MAPPING = {
    "inventory_full":  cv2.cvtColor(cv2.imread(r"C:\Projects\image-detection\match_images\inventory_full.JPG"), cv2.COLOR_BGR2GRAY),
    "click_here_inventory_full": np.asarray(Image.open(r"C:\Projects\image-detection\match_images\click_here.JPG"))
}

# full runelite client
CLIENT_COORDINATES = {"top": 70, "left": 5, "width": 765, "height": 505}

# frames per second
FPS = 5

# Weight File
WEIGHTS = r'C:\Projects\image-detection\runs\detect\train8\weights\best.pt'

# model
model = YOLO(WEIGHTS)

mouse = Controller()
keyboard = Kontroller()

# setting up sscreen shotting
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
        
        # Do this before we do anything
        if frame_number % 4 == 0:
            is_inventory_full(
                gray_copy,
                STATIC_IMAGE_MAPPING["inventory_full"],
                mouse,
                keyboard
            )
        
        # iterate over results
        for result in results:

            # Get bounding boxes
            for box in result.boxes:
                b = box.xyxy[0]
                f = box.xyxyn[0]
                c = box.cls

                # Logic for type of record (110 = tin)
                if "110" in result.names[int(c)]:
                    # print(f)
                    # print(b[0])
                    click_point = midpoint(b[0], b[1], b[2], b[3])
                    print(click_point)
                    print(result.names[int(c)])
                    print("TIN SPOTTED")
                    
                    click_on_screen(mouse, click_point, True)
                    break
                    # Application logic here
                    


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