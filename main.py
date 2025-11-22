import cv2
import numpy as np

WIDTH = 1280
HEIGHT = 720
offset = (WIDTH - HEIGHT) // 2
CELL_SIDE = HEIGHT // 8
board_state = np.array([
    [ 0,  1,  0,  1,  0,  1,  0,  1],
    [ 1,  0,  1,  0,  1,  0,  1,  0],
    [ 0,  1,  0,  1,  0,  1,  0,  1],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [-1,  0, -1,  0, -1,  0, -1,  0],
    [ 0, -1,  0, -1,  0, -1,  0, -1],
    [-1,  0, -1,  0, -1,  0, -1,  0]
], dtype=int)

board = cv2.imread("assets/board.png", cv2.IMREAD_COLOR)
board = cv2.multiply(board, 0.5)
red_piece = cv2.imread("assets/red_piece.png", cv2.IMREAD_UNCHANGED)
red_piece = cv2.resize(red_piece, (CELL_SIDE, CELL_SIDE))
black_piece = cv2.imread("assets/black_piece.png", cv2.IMREAD_UNCHANGED)
black_piece = cv2.resize(black_piece, (CELL_SIDE, CELL_SIDE))


def getCentroid(mask):
    points = np.column_stack(np.where(mask > 0))
    if len(points) == 0:
        return None
    (center), (w, h), angle = cv2.minAreaRect(points)
    return (int(center[1]), int(center[0]))


def drawBoard(frame):
    overlay = cv2.add(frame, board)
    for i, row in enumerate(board_state):
        for j, value in enumerate(row):
            if value == 1:
                piece = red_piece
            elif value == -1:
                piece = black_piece
            else:
                continue

            h, w, _ = piece.shape
            y1, y2 = i * CELL_SIDE, i * CELL_SIDE + h
            x1, x2 = j * CELL_SIDE, j * CELL_SIDE + w

            piece_rgb = piece[:, :, :3]
            alpha = piece[:, :, 3] / 255.0
            roi = overlay[y1:y2, x1:x2]

            for c in range(3):
                roi[:, :, c] = roi[:, :, c] * (1 - alpha) + piece_rgb[:, :, c] * alpha
            overlay[y1:y2, x1:x2] = roi
    return overlay


webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

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

    if not ret:
        print("Error: Could not read frame.")
        break
    
    frame = frame[:, offset:(WIDTH - offset)]
    frame = cv2.flip(frame, 1)
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_frame)

    blue_mask = cv2.threshold(b, 90, 255, cv2.THRESH_BINARY_INV)[1]
    red_mask = cv2.threshold(a, 200, 255, cv2.THRESH_BINARY)[1]
    
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel=kernel_close)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel=kernel_close)

    blue_centroid = getCentroid(blue_mask)
    red_centroid = getCentroid(red_mask)

    frame = drawBoard(frame)
    if blue_centroid:
        frame = cv2.circle(frame, center=blue_centroid, radius=10, color=(255, 0, 0), thickness=5)
    if red_centroid:
        frame = cv2.circle(frame, center=red_centroid, radius=10, color=(0, 0, 255), thickness=5)

    cv2.imshow('red', red_mask)
    cv2.imshow('blue', blue_mask)
    cv2.imshow('webcam', frame)

    key = cv2.waitKey(33) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()