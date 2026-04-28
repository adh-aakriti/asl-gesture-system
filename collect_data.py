import cv2
import mediapipe as mp
import os
import numpy as np
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ====== CONFIG ======
DATA_DIR = "data_custom"
labels = ["yes", "no", "wait", "help", "thank_you", "stop", "emergency"]

# Create folders
for label in labels:
    os.makedirs(os.path.join(DATA_DIR, label), exist_ok=True)

# Load MediaPipe model
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

current_label = "yes"
flat_landmarks = None  # IMPORTANT: avoid undefined variable error

print("\nControls:")
print("1 = yes | 2 = no | 3 = wait | 4 = help | 5 = thank_you")
print("6 = stop | 7 = emergency")
print("s = save sample | q = quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue  # safer than break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = detector.detect(mp_image)

    flat_landmarks = None  # reset every frame

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:

            landmarks = []

            for lm in hand_landmarks:
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                landmarks.append([lm.x, lm.y, lm.z])

            flat_landmarks = np.array(landmarks).flatten()

    # -----------------------
    # COUNT FILES (REAL COUNT)
    # -----------------------
    current_folder = os.path.join(DATA_DIR, current_label)
    count = len(os.listdir(current_folder))

    # Display label + count
    cv2.putText(frame, f"Label: {current_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Saved: {count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Collect Data", frame)

    key = cv2.waitKey(1) & 0xFF

    # Switch labels
    if key == ord('1'): current_label = "yes"
    elif key == ord('2'): current_label = "no"
    elif key == ord('3'): current_label = "wait"
    elif key == ord('4'): current_label = "help"
    elif key == ord('5'): current_label = "thank_you"
    elif key == ord('6'): current_label = "stop"
    elif key == ord('7'): current_label = "emergency"

    # Save sample
    elif key == ord('s'):
        if flat_landmarks is not None:
            file_name = f"{int(time.time() * 1000)}.npy"
            file_path = os.path.join(DATA_DIR, current_label, file_name)

            np.save(file_path, flat_landmarks)

            print(f"Saved {current_label} sample")
        else:
            print("No hand detected — not saved")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()