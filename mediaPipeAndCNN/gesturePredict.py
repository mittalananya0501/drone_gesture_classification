from pathlib import Path
import time
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "gesture_cnn.h5"
LABELS_PATH = PROJECT_ROOT / "model" / "labels.npy"
FALLBACK_CLASS_NAMES = np.array(
    ["DOWN", "FLIP", "LAND", "LEFT", "RIGHT", "TAKEOFF", "UP"]
)


def open_camera(preferred_indices=(1, 0)):
    for index in preferred_indices:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            return cap, index
        cap.release()
    raise RuntimeError(
        "Could not open a camera. Try checking camera permissions or using a different index."
    )


if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = load_model(MODEL_PATH)
class_names = np.load(LABELS_PATH, allow_pickle=True) if LABELS_PATH.exists() else FALLBACK_CLASS_NAMES

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap, camera_index = open_camera()

fingertips = [4, 8, 12, 16, 20]

prev_gesture = None
start_time = None
command_triggered = False
HOLD_TIME = 5

while True:
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame from camera index {camera_index}.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data = []
            for landmark_id in fingertips:
                lm = hand_landmarks.landmark[landmark_id]
                data.append(lm.x)
                data.append(lm.y)

            X = np.array(data, dtype=np.float32).reshape(1, 10, 1)
            prediction = model.predict(X, verbose=0)
            gesture_index = int(np.argmax(prediction))
            gesture_name = str(class_names[gesture_index])

            if gesture_name == prev_gesture:
                if start_time and time.time() - start_time > HOLD_TIME and not command_triggered:
                    print("Command Triggered:", gesture_name)
                    command_triggered = True
            else:
                prev_gesture = gesture_name
                start_time = time.time()
                command_triggered = False

            cv2.putText(
                frame,
                gesture_name,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                5,
            )

    else:
        prev_gesture = None
        start_time = None
        command_triggered = False

    cv2.imshow("Prediction", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
