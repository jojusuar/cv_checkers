import cv2
import numpy as np
from BoardState import BoardState

WIDTH = 1280
HEIGHT = 720
offset = (WIDTH - HEIGHT) // 2

def getCentroid(mask):
    points = np.column_stack(np.where(mask > 0))
    if len(points) == 0:
        return None
    (center), (w, h), angle = cv2.minAreaRect(points)
    return (int(center[1]), int(center[0]))

board_state = BoardState(shape=(HEIGHT, HEIGHT))

webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

cv2.namedWindow('blue', cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow('red', cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow('webcam', cv2.WINDOW_GUI_NORMAL)
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

while True:
    ret, frame = webcam.read()

    if not ret:
        print("Error: Could not read frame.")
        break
    
    frame = frame[:, offset:(WIDTH - offset)]
    frame = cv2.flip(frame, 1)
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_frame)

    blue_mask = cv2.threshold(b, 70, 255, cv2.THRESH_BINARY_INV)[1]
    red_mask = cv2.threshold(a, 200, 255, cv2.THRESH_BINARY)[1]
    
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel=kernel_close)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel=kernel_close)

    blue_centroid = getCentroid(blue_mask)
    red_centroid = getCentroid(red_mask)

    frame = board_state.drawBoard(frame=frame,
                                  blue_centroid=blue_centroid,
                                  red_centroid=red_centroid)

    cv2.imshow('red', red_mask)
    cv2.imshow('blue', blue_mask)
    cv2.imshow('webcam', frame)

    key = cv2.waitKey(33) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()