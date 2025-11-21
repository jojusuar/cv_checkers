import cv2
import numpy as np

def getCentroid(mask):
    points = np.column_stack(np.where(mask > 0))
    if len(points) == 0:
        return None
    (center), (w, h), angle = cv2.minAreaRect(points)
    return (int(center[1]), int(center[0]))

def drawBoard(frame, board):
    overlay = cv2.add(frame, board)
    return overlay

WIDTH = 1280
HEIGHT = 720
offset = (WIDTH - HEIGHT) // 2

webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

board = cv2.imread("assets/board.png", cv2.IMREAD_COLOR)

ret, frame = webcam.read()
frame = frame[:, offset:(WIDTH - offset)]
width, height, channels = frame.shape
board = cv2.resize(board, (height, width))

cv2.namedWindow('blue', cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow('red', cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow('webcam', cv2.WINDOW_GUI_NORMAL)
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

while True:
    ret, frame = webcam.read()
    frame = frame[:, offset:(WIDTH - offset)]

    if not ret:
        print("Error: Could not read frame.")
        break
    
    frame = cv2.flip(frame, 1)
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_frame)

    blue_mask = cv2.threshold(b, 90, 255, cv2.THRESH_BINARY_INV)[1]
    red_mask = cv2.threshold(a, 180, 255, cv2.THRESH_BINARY)[1]
    
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel=kernel_close)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel=kernel_close)

    frame = drawBoard(frame, board)

    blue_centroid = getCentroid(blue_mask)
    red_centroid = getCentroid(red_mask)

    if blue_centroid:
        frame = cv2.circle(frame, center=blue_centroid, radius=5, color=(255, 0, 0), thickness=2)
    if red_centroid:
        frame = cv2.circle(frame, center=red_centroid, radius=5, color=(0, 0, 255), thickness=2)

    cv2.imshow('red', red_mask)
    cv2.imshow('blue', blue_mask)
    cv2.imshow('webcam', frame)

    key = cv2.waitKey(33) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()