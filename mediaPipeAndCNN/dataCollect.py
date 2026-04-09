from pathlib import Path
import csv

import cv2
import mediapipe as mp


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
MAX_SAMPLES = 300
FINGERTIPS = [4, 8, 12, 16, 20]


def open_camera(preferred_indices=(1, 0)):
    for index in preferred_indices:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            return cap, index
        cap.release()
    raise RuntimeError(
        "Could not open a camera. Try checking camera permissions or using a different index."
    )


mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap, camera_index = open_camera()

sample_count = 0

gesture_name = input("Enter gesture name: ").strip()
if not gesture_name:
    raise ValueError("Gesture name cannot be empty.")

DATASET_DIR.mkdir(exist_ok=True)
save_path = DATASET_DIR / f"{gesture_name}.csv"

print(f"Saving samples to {save_path}")
print(f"Using camera index {camera_index}")
print("Press 's' to save data. Press ESC to quit.")

with open(save_path, mode="a", newline="") as file:
    writer = csv.writer(file)

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame from camera index {camera_index}.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)
        current_sample = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data = []

                for landmark_id in FINGERTIPS:
                    lm = hand_landmarks.landmark[landmark_id]
                    data.append(lm.x)
                    data.append(lm.y)

                current_sample = data
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                break

        cv2.putText(
            frame,
            f"Samples: {sample_count}/{MAX_SAMPLES}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "Press 's' to save, ESC to quit",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            if current_sample is None:
                print("No hand detected. Show your hand before saving.")
            else:
                writer.writerow(current_sample)
                sample_count += 1
                print(f"Saved: {sample_count}")

        if sample_count >= MAX_SAMPLES:
            print(f"Collected {MAX_SAMPLES} samples!")
            break

        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()
