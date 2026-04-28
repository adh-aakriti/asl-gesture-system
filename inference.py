import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
import threading
import time
import os
import urllib.request

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------------------
# DOWNLOAD MODEL IF NEEDED
# -----------------------
MODEL_FILE = "hand_landmarker.task"
if not os.path.exists(MODEL_FILE):
    print("Downloading hand_landmarker.task...")
    # Try multiple sources
    urls = [
        "https://ai.google.dev/static/mediapipe/hand_landmarker.task",
        "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task",
    ]
    
    for url in urls:
        try:
            urllib.request.urlretrieve(url, MODEL_FILE)
            print(f"✓ Downloaded from {url}")
            break
        except Exception as e:
            print(f"✗ {url}: {e}")
            continue
    else:
        print(f"⚠ Could not download model file. Please download hand_landmarker.task manually")
        print(f"  Save it to: {os.path.abspath(MODEL_FILE)}")
        exit(1)

# -----------------------
# LOAD MODEL
# -----------------------
model = joblib.load("model_custom.pkl")

# -----------------------
# MEDIAPIPE SETUP
# -----------------------
base_options = python.BaseOptions(model_asset_path=MODEL_FILE)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

# -----------------------
# TEXT TO SPEECH
# -----------------------
tts_engine = None

def initialize_tts():
    """Lazy initialize TTS engine"""
    global tts_engine
    if tts_engine is None:
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 150)  # Speed of speech

def speak_async(text):
    """Speak text in a separate thread (non-blocking)"""
    try:
        initialize_tts()
        thread = threading.Thread(target=lambda: tts_engine.say(text) or tts_engine.runAndWait())
        thread.daemon = True
        thread.start()
    except Exception as e:
        print(f"Speech error: {e}")

# -----------------------
# SMOOTHING & GESTURE TRACKING
# -----------------------
history = []
display_text = ""
current_gesture = None
gesture_start_time = None
gesture_spoken = False
HOLD_TIME_SECONDS = 1.0  # Time to hold gesture before speaking

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
            # AUTO-SPEAK WITH HOLD TIME
            # -----------------------
            if display_text and history.count(final_prediction) >= 5:
                if current_gesture != display_text:
                    # New gesture detected
                    current_gesture = display_text
                    gesture_start_time = time.time()
                    gesture_spoken = False
                else:
                    # Same gesture, check hold time
                    hold_duration = time.time() - gesture_start_time
                    
                    if hold_duration >= HOLD_TIME_SECONDS and not gesture_spoken:
                        # Gesture held long enough, speak it
                        speak_async(display_text)
                        gesture_spoken = True
            else:
                # No valid gesture detected
                current_gesture = None
                gesture_start_time = None
                gesture_spoken = False

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

    # Show hold progress
    if current_gesture and gesture_start_time and not gesture_spoken:
        hold_duration = time.time() - gesture_start_time
        progress = min(hold_duration / HOLD_TIME_SECONDS, 1.0)
        cv2.putText(
            frame,
            f"Holding: {progress:.0%}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
    elif gesture_spoken:
        cv2.putText(
            frame,
            "Spoken!",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

    cv2.imshow("ASL Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    # -----------------------
    # CONTROLS
    # -----------------------

    # quit
    if key == ord('q'):
        break

    # manually speak current gesture (if auto-speak hasn't triggered)
    elif key == ord('s'):
        if display_text and not gesture_spoken:
            speak_async(display_text)
            gesture_spoken = True
    
    # reset gesture tracking (press 'r')
    elif key == ord('r'):
        current_gesture = None
        gesture_start_time = None
        gesture_spoken = False

cap.release()
cv2.destroyAllWindows()