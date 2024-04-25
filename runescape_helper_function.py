from pynput.mouse import Button
from pynput.keyboard import Key
import time
import cv2
import numpy as np


POSITIONS = {
    "inventory_bag": (661, 257),
    "first_item_first_row": (582, 298),
    "second_item_first_row": (624, 298),
    "third_item_first_row": (668, 298),
    "last_item_first_row": (711, 298),
}

LIST_POSITIONS = [
    (582, 298),
    (624, 298),
    (668, 298),
    (711, 298)
]

def midpoint(x1, y1, x2, y2):
    return ((x1 + x2)/2, (y1 + y2)/2)

# def click_on_screen(mouse, position):
#     mouse.position = position
#     mouse.press(Button.left)
#     mouse.release(Button.left)

def click_on_screen(mouse, position, offset=False, click=True):
    ''''''
    if offset:
        y = list(position)
        print("Original Coordinates: ", str(y))
        y[0] = position[0] + 10
        y[1] = position[1] + 60
        position = tuple(y)
        print("Fixed Coordinates: ", str(position))

    mouse.position = position
    time.sleep(.5)
    if click:
        mouse.press(Button.left)
        mouse.release(Button.left)


def is_inventory_full(base_image, lookup_image, mouse, keyboard):
    if check_if_match(base_image, lookup_image):
        empty_inventory(mouse, keyboard)

def check_if_match(base_image, lookup_image):
    """"""
    W, H = lookup_image.shape[:2]
    
    res = cv2.matchTemplate(base_image,lookup_image,cv2.TM_CCOEFF)
    # Define a minimum threshold
    thresh = 6000000

    boxes = list()

    # Select rectangles with confidence greater than threshold
    (y_points, x_points) = np.where(res >= thresh)
    
    for (x, y) in zip(x_points, y_points):
        # update our list of rectangles
        boxes.append((x, y, x + W, y + H))
    
    if len(boxes) > 0:
        return True

    return False



def empty_inventory(mouse, keyboard):
    """"""
    print("Emptying inventory")
    click_on_screen(
        mouse,
        POSITIONS["inventory_bag"],
        False,
        True
    )
    
    keyboard.press(Key.shift_l)
    for record in LIST_POSITIONS:
        click_on_screen(
            mouse, record, False, True
        )
        time.sleep(.5)
    
    keyboard.release(Key.shift_l)
    

    print("Inventory Emptied")