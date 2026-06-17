# extract_landmarks_from_dataset.py
# ------------------------------------------------------------
# This file reads the downloaded ASL alphabet image dataset,
# extracts hand landmarks using MediaPipe, and saves them into
# a CSV file called landmarks.csv.
#
# Instead of training directly on images using CNN, we convert
# each hand image into 21 hand landmark points.
#
# Each hand has 21 points, and each point has x, y, z values.
# So one image becomes:
# 21 * 3 = 63 numerical values
# ------------------------------------------------------------

import cv2
import csv
from pathlib import Path
import mediapipe as mp
from labels import LABELS


# ------------------------------------------------------------
# Step 1: Define project paths
# ------------------------------------------------------------

# ROOT_DIR points to the main project folder: sign_lingo_alphabet/
ROOT_DIR = Path(__file__).resolve().parents[1]

# DATASET_DIR points to the downloaded training dataset folder
# Expected folder:
# sign_lingo_alphabet/dataset/asl_alphabet_train/
DATASET_DIR = ROOT_DIR / "dataset" / "asl_alphabet_train"

# OUTPUT_CSV is where extracted landmark values will be saved
# Final file:
# sign_lingo_alphabet/dataset/landmarks.csv
OUTPUT_CSV = ROOT_DIR / "dataset" / "landmarks.csv"


# ------------------------------------------------------------
# Step 2: Limit images per class
# ------------------------------------------------------------

# The ASL dataset has many images per class.
# To save time, we use only 500 images from each class first.
# Later, if needed, you can increase this to 1000 or more.
MAX_IMAGES_PER_CLASS = 500


# ------------------------------------------------------------
# Step 3: Initialize MediaPipe Hands
# ------------------------------------------------------------

# mp_hands gives access to MediaPipe's hand detection model
mp_hands = mp.solutions.hands


# ------------------------------------------------------------
# Step 4: Function to normalize landmarks
# ------------------------------------------------------------

def normalize_landmarks(hand_landmarks):
    """
    This function converts MediaPipe hand landmarks into normalized values.

    Why normalization is needed:
    ----------------------------
    Hand position may change in different images.
    Sometimes the hand is left, right, near, or far from the camera.

    To make the model focus on the hand shape instead of hand position,
    we subtract the wrist point from all other points.

    Wrist point is landmark 0.
    """

    # Get all 21 landmarks from MediaPipe
    landmarks = hand_landmarks.landmark

    # Landmark 0 is the wrist
    wrist = landmarks[0]

    # This list will store normalized x, y, z values
    values = []

    # Go through all 21 landmarks
    for lm in landmarks:
        # Make wrist the reference point
        # This removes hand location effect
        x = lm.x - wrist.x
        y = lm.y - wrist.y
        z = lm.z - wrist.z

        # Add x, y, z values to the list
        values.extend([x, y, z])

    # Find the largest absolute value
    # This helps scale all values into a similar range
    max_value = max(abs(v) for v in values)

    # Avoid division by zero
    if max_value == 0:
        max_value = 1

    # Scale all landmark values
    values = [v / max_value for v in values]

    return values


# ------------------------------------------------------------
# Step 5: Main function
# ------------------------------------------------------------

def main():
    # Check whether dataset folder exists
    if not DATASET_DIR.exists():
        print("Dataset folder not found!")
        print("Expected path:", DATASET_DIR)
        return

    # Create dataset folder if it does not already exist
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Create CSV header
    # First column is label, remaining columns are x0,y0,z0 ... x20,y20,z20
    header = ["label"]

    for i in range(21):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])

    # These variables count total saved and skipped images
    total_saved = 0
    total_skipped = 0

    # Open CSV file in write mode
    # newline="" avoids blank rows in Windows
    with open(OUTPUT_CSV, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write header row into CSV
        writer.writerow(header)

        # Start MediaPipe Hands model
        # static_image_mode=True because we are processing images, not video
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        ) as hands:

            # Process each class label one by one
            # Example: A, B, C, ..., Z, del, nothing, space
            for label in LABELS:

                # Path of current class folder
                # Example:
                # dataset/asl_alphabet_train/A
                class_folder = DATASET_DIR / label

                # If folder does not exist, skip that label
                if not class_folder.exists():
                    print(f"Folder not found for label: {label}")
                    continue

                # Get image files from the current class folder
                # Only first MAX_IMAGES_PER_CLASS images are used
                image_files = list(class_folder.glob("*"))[:MAX_IMAGES_PER_CLASS]

                print(f"\nProcessing label: {label}")
                print(f"Images selected: {len(image_files)}")

                # Count saved and skipped samples for this label
                saved_count = 0
                skipped_count = 0

                # Process every image in the selected class folder
                for image_path in image_files:

                    # Read image using OpenCV
                    image = cv2.imread(str(image_path))

                    # If image cannot be read, skip it
                    if image is None:
                        skipped_count += 1
                        continue

                    # Special case for "nothing"
                    # "nothing" means no hand/sign action.
                    # So we save 63 zeros instead of hand landmarks.
                    if label == "nothing":
                        writer.writerow([label] + [0] * 63)
                        saved_count += 1
                        continue

                    # Convert image from BGR to RGB
                    # OpenCV reads images in BGR format,
                    # but MediaPipe expects RGB format.
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Detect hand landmarks using MediaPipe
                    result = hands.process(rgb_image)

                    # If hand is detected
                    if result.multi_hand_landmarks:

                        # Take the first detected hand
                        hand_landmarks = result.multi_hand_landmarks[0]

                        # Normalize landmark points
                        landmark_values = normalize_landmarks(hand_landmarks)

                        # Write label and 63 landmark values into CSV
                        writer.writerow([label] + landmark_values)

                        saved_count += 1

                    else:
                        # If MediaPipe could not detect hand, skip that image
                        skipped_count += 1

                # Update total counts
                total_saved += saved_count
                total_skipped += skipped_count

                # Print result for current label
                print(f"Saved: {saved_count}")
                print(f"Skipped: {skipped_count}")

    # Final summary
    print("\nLandmark extraction completed.")
    print("CSV saved at:", OUTPUT_CSV)
    print("Total saved samples:", total_saved)
    print("Total skipped images:", total_skipped)


# ------------------------------------------------------------
# Step 6: Run the main function
# ------------------------------------------------------------

if __name__ == "__main__":
    main()