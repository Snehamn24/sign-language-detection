# predict_sentence.py
# ------------------------------------------------------------
# Sign Lingo - Fast Real-Time Sign Prediction and Sentence Builder
#
# This version is made for demo:
# 1. Prediction is shown continuously
# 2. Letter gets added quickly when stable
# 3. You can press 's' to manually add current prediction
# 4. You can press 'c' to clear sentence
# 5. You can press 'q' to quit
#
# A-Z      -> append letter
# space    -> add blank space
# del      -> delete last character
# nothing  -> no action
# ------------------------------------------------------------

import cv2
import time
import joblib
import pandas as pd
from pathlib import Path
import mediapipe as mp


# ------------------------------------------------------------
# Step 1: Define paths
# ------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "sign_lingo_model.pkl"


# ------------------------------------------------------------
# Step 2: Settings for fast sentence formation
# ------------------------------------------------------------

# Lower confidence because webcam lighting may differ from dataset images
CONFIDENCE_THRESHOLD = 0.60

# Lower stable frames means faster accepting
STABLE_FRAMES_REQUIRED = 3

# Minimum gap between two accepted letters
COOLDOWN_SECONDS = 0.35

# If same sign is held continuously, add it again only after this time
# Useful for repeated letters like LL, OO, EE
SAME_SIGN_REPEAT_SECONDS = 1.2


# ------------------------------------------------------------
# Step 3: MediaPipe setup
# ------------------------------------------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# ------------------------------------------------------------
# Step 4: Feature column names
# ------------------------------------------------------------

FEATURE_COLUMNS = []

for i in range(21):
    FEATURE_COLUMNS.extend([f"x{i}", f"y{i}", f"z{i}"])


# ------------------------------------------------------------
# Step 5: Normalize landmarks
# ------------------------------------------------------------

def normalize_landmarks(hand_landmarks):
    """
    Converts MediaPipe hand landmarks into 63 normalized values.

    This same method was used during landmark extraction from dataset.
    """

    landmarks = hand_landmarks.landmark
    wrist = landmarks[0]

    values = []

    for lm in landmarks:
        x = lm.x - wrist.x
        y = lm.y - wrist.y
        z = lm.z - wrist.z

        values.extend([x, y, z])

    max_value = max(abs(v) for v in values)

    if max_value == 0:
        max_value = 1

    values = [v / max_value for v in values]

    return values


# ------------------------------------------------------------
# Step 6: Update sentence
# ------------------------------------------------------------

def update_sentence(sentence, label):
    """
    Updates sentence based on predicted label.
    """

    if label == "space":
        sentence += " "

    elif label == "del":
        sentence = sentence[:-1]

    elif label == "nothing":
        pass

    else:
        sentence += label

    return sentence


# ------------------------------------------------------------
# Step 7: Main function
# ------------------------------------------------------------

def main():
    # Check model file
    if not MODEL_PATH.exists():
        print("Model not found!")
        print("Expected path:", MODEL_PATH)
        return

    # Load trained Random Forest model
    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully.")
    print("Controls:")
    print("q -> quit")
    print("c -> clear sentence")
    print("s -> manually add current prediction")
    print("space key -> add keyboard space")
    print("backspace key -> delete last character")

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    # Sentence output
    sentence = ""

    # Current prediction
    predicted_label = "nothing"
    confidence = 0.0

    # Stable prediction tracking
    candidate_label = "nothing"
    candidate_count = 0

    # Last accepted tracking
    last_added_label = None
    last_added_time = 0

    # Message shown on screen
    status_text = "Show a sign"

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Mirror view
            frame = cv2.flip(frame, 1)

            # Convert frame to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect hand
            result = hands.process(rgb_frame)

            # Default values
            predicted_label = "nothing"
            confidence = 0.0

            # ------------------------------------------------
            # Prediction section
            # ------------------------------------------------

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # Extract landmark features
                landmark_values = normalize_landmarks(hand_landmarks)

                input_data = pd.DataFrame(
                    [landmark_values],
                    columns=FEATURE_COLUMNS
                )

                # Predict probabilities
                probabilities = model.predict_proba(input_data)[0]

                # Highest probability class
                max_index = probabilities.argmax()

                predicted_label = model.classes_[max_index]
                confidence = probabilities[max_index]

            # ------------------------------------------------
            # Stable prediction logic
            # ------------------------------------------------

            # Count how many continuous frames have same prediction
            if predicted_label == candidate_label:
                candidate_count += 1
            else:
                candidate_label = predicted_label
                candidate_count = 1

            current_time = time.time()

            # ------------------------------------------------
            # Fast automatic sentence appending
            # ------------------------------------------------

            can_add = False

            if (
                predicted_label != "nothing"
                and confidence >= CONFIDENCE_THRESHOLD
                and candidate_count >= STABLE_FRAMES_REQUIRED
                and current_time - last_added_time >= COOLDOWN_SECONDS
            ):
                # If new sign is different from last added, allow quickly
                if predicted_label != last_added_label:
                    can_add = True

                # If same sign is held, allow only after delay
                elif current_time - last_added_time >= SAME_SIGN_REPEAT_SECONDS:
                    can_add = True

            if can_add:
                sentence = update_sentence(sentence, predicted_label)

                last_added_label = predicted_label
                last_added_time = current_time

                status_text = f"Added: {predicted_label}"

                print("Added:", predicted_label)
                print("Sentence:", sentence)

            else:
                if predicted_label == "nothing":
                    status_text = "Show a sign"
                    last_added_label = None
                elif confidence < CONFIDENCE_THRESHOLD:
                    status_text = "Low confidence"
                elif candidate_count < STABLE_FRAMES_REQUIRED:
                    status_text = "Hold steady"
                else:
                    status_text = "Ready / change sign"

            # ------------------------------------------------
            # Display on screen
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Prediction: {predicted_label} ({confidence:.2f})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Stable Frames: {candidate_count}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                status_text,
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Sentence:",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            # Show full sentence in terminal, last part on screen
            cv2.putText(
                frame,
                sentence[-40:],
                (20, 215),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "q: Quit | c: Clear | s: Add | space: Space",
                (20, 460),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.imshow("Sign Lingo - Fast Sentence Builder", frame)

            # ------------------------------------------------
            # Keyboard controls
            # ------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("c"):
                sentence = ""
                last_added_label = None
                last_added_time = 0
                status_text = "Sentence cleared"
                print("Sentence cleared")

            elif key == ord("s"):
                # Manual add current prediction
                if predicted_label != "nothing" and confidence >= CONFIDENCE_THRESHOLD:
                    sentence = update_sentence(sentence, predicted_label)
                    last_added_label = predicted_label
                    last_added_time = time.time()

                    print("Manually added:", predicted_label)
                    print("Sentence:", sentence)

            elif key == 32:
                # Keyboard space key
                sentence += " "
                print("Keyboard space added")
                print("Sentence:", sentence)

            elif key == 8:
                # Backspace key
                sentence = sentence[:-1]
                print("Deleted last character")
                print("Sentence:", sentence)

    cap.release()
    cv2.destroyAllWindows()


# ------------------------------------------------------------
# Step 8: Run program
# ------------------------------------------------------------

if __name__ == "__main__":
    main()