import cv2
import mediapipe as mp
import numpy as np
from BoardState import BoardState
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2,
                                       running_mode=vision.RunningMode.VIDEO)
hands = vision.HandLandmarker.create_from_options(options)

WIDTH = 1280
HEIGHT = 720
offset = (WIDTH - HEIGHT) // 2

def unsharp_masking(image, sigma=1.0, strength=0.5):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened

board_state = BoardState(shape=(HEIGHT, HEIGHT))

# Módulo de captura de imagen
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

cv2.namedWindow('webcam', cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow('processed', cv2.WINDOW_GUI_NORMAL)
while True:
    ret, frame = webcam.read()

    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Módulo de detección de posición
    frame = frame[:, offset:(WIDTH - offset)]
    frame = cv2.flip(frame, 1)

    # Preprocesamiento: mitigación de ruido y realzando de contornos
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab_clahe = cv2.merge([l, a, b])
    frame = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    denoised = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
    to_process = unsharp_masking(denoised, sigma=1.0, strength=3)
    to_process = cv2.cvtColor(to_process, cv2.COLOR_BGR2RGB)

    # Detección de landmarks
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=to_process)
    timestamp_ms = int(webcam.get(cv2.CAP_PROP_POS_MSEC))
    results = hands.detect_for_video(mp_image, timestamp_ms)
    
    thumb_landmark = None
    index_landmark = None
    if results.hand_landmarks:
        hand = results.hand_landmarks[0]
        thumb_landmark = hand[4]
        index_landmark = hand[8]
        
    # Renderizado del tablero
    frame = board_state.drawBoard(frame=frame,
                                thumb=thumb_landmark,
                                index=index_landmark)

    cv2.imshow('webcam', frame)
    key = cv2.waitKey(round(1000/30)) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()