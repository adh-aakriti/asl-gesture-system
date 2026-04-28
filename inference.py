import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------------------
# LOAD MODEL
# -----------------------
model = joblib.load("model_custom.pkl")

# -----------------------
# MEDIAPIPE SETUP
# -----------------------
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

# -----------------------
# TEXT TO SPEECH
# -----------------------
engine = pyttsx3.init()

def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except:
        print("Speech error")

# -----------------------
# SMOOTHING
# -----------------------
history = []
display_text = ""

# -----------------------
# WEBCAM
# -----------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:

            landmarks = []

            for lm in hand_landmarks:
                landmarks.append([lm.x, lm.y, lm.z])

            flat = np.array(landmarks).flatten().reshape(1, -1)

            prediction = model.predict(flat)[0]

            # -----------------------
            # SMOOTHING
            # -----------------------
            history.append(prediction)

            if len(history) > 10:
                history.pop(0)

            final_prediction = max(set(history), key=history.count)

            if history.count(final_prediction) >= 5:
                display_text = final_prediction
            else:
                display_text = ""

    # -----------------------
    # UI (SUBTITLES)
    # -----------------------
    h, w, _ = frame.shape

    # black bar
    cv2.rectangle(frame, (0, h - 80), (w, h), (0, 0, 0), -1)

    # subtitle text
    cv2.putText(
        frame,
        display_text.upper(),
        (50, h - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (255, 255, 255),
        3
    )

    cv2.imshow("ASL Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    # -----------------------
    # CONTROLS
    # -----------------------

    # quit
    if key == ord('q'):
        break

    # speak current word
    elif key == ord('s'):
        if display_text != "":
            speak(display_text)

cap.release()
cv2.destroyAllWindows()