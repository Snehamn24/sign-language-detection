# project_statistics.py
# ------------------------------------------------------------
# This file generates complete statistics for the Sign Lingo project.
#
# It gives:
# 1. Dataset statistics
# 2. Class-wise sample count
# 3. Train-test split count
# 4. Model accuracy
# 5. Classification report
# 6. Confusion matrix image
# 7. Text report for presentation/documentation
# ------------------------------------------------------------

import pandas as pd
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ------------------------------------------------------------
# Step 1: Define project paths
# ------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = ROOT_DIR / "dataset" / "asl_alphabet_train"
CSV_PATH = ROOT_DIR / "dataset" / "landmarks.csv"
MODEL_PATH = ROOT_DIR / "models" / "sign_lingo_model.pkl"

REPORTS_DIR = ROOT_DIR / "reports"
REPORT_TEXT_PATH = REPORTS_DIR / "project_statistics.txt"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"


# Same value used during landmark extraction
MAX_IMAGES_PER_CLASS = 500


# ------------------------------------------------------------
# Step 2: Main function
# ------------------------------------------------------------

def main():
    # Create reports folder if it does not exist
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Check required files
    if not CSV_PATH.exists():
        print("landmarks.csv not found!")
        return

    if not MODEL_PATH.exists():
        print("Trained model not found!")
        return

    # Load landmark dataset
    data = pd.read_csv(CSV_PATH)

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Separate input features and output labels
    X = data.drop("label", axis=1)
    y = data["label"]

    # Recreate the same train-test split used during training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Predict test data
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Count samples per class
    sample_counts = data["label"].value_counts().sort_index()

    # Dataset folder statistics
    class_folders = [folder for folder in DATASET_DIR.iterdir() if folder.is_dir()]
    total_classes = len(class_folders)

    # Count selected images based on MAX_IMAGES_PER_CLASS
    total_selected_images = 0

    for folder in class_folders:
        image_count = len(list(folder.glob("*")))
        selected_count = min(image_count, MAX_IMAGES_PER_CLASS)
        total_selected_images += selected_count

    total_landmark_samples = len(data)
    total_skipped_images = total_selected_images - total_landmark_samples

    # Generate classification report
    report = classification_report(y_test, y_pred)

    # --------------------------------------------------------
    # Step 3: Print statistics in terminal
    # --------------------------------------------------------

    print("\n========== PROJECT STATISTICS ==========")

    print("\nProject Name:")
    print("Sign Lingo: Real-Time Sign Language Alphabet Recognition and Sentence Formation")

    print("\nDataset:")
    print("ASL Alphabet Dataset from Kaggle")

    print("\nApproach:")
    print("MediaPipe Hands + Random Forest Classifier")

    print("\nTotal Classes:", total_classes)
    print("Total selected images:", total_selected_images)
    print("Total landmark samples saved:", total_landmark_samples)
    print("Total skipped images:", total_skipped_images)

    print("\nInput Features:")
    print("21 hand landmarks × 3 coordinates = 63 features")

    print("\nTrain-Test Split:")
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print("\nModel Accuracy:", accuracy)

    print("\nSamples Per Class:")
    print(sample_counts)

    print("\nClassification Report:")
    print(report)

    # --------------------------------------------------------
    # Step 4: Save text report
    # --------------------------------------------------------

    with open(REPORT_TEXT_PATH, "w") as file:
        file.write("SIGN LINGO PROJECT STATISTICS\n")
        file.write("====================================\n\n")

        file.write("Project Name:\n")
        file.write("Sign Lingo: Real-Time Sign Language Alphabet Recognition and Sentence Formation\n\n")

        file.write("Dataset Used:\n")
        file.write("ASL Alphabet Dataset from Kaggle\n\n")

        file.write("Approach Used:\n")
        file.write("MediaPipe Hands + Random Forest Classifier\n\n")

        file.write("Dataset Statistics:\n")
        file.write(f"Total classes: {total_classes}\n")
        file.write(f"Total selected images: {total_selected_images}\n")
        file.write(f"Total landmark samples saved: {total_landmark_samples}\n")
        file.write(f"Total skipped images: {total_skipped_images}\n\n")

        file.write("Feature Details:\n")
        file.write("Each hand image is converted into 21 hand landmarks.\n")
        file.write("Each landmark has x, y, and z coordinates.\n")
        file.write("Total features per sample = 21 × 3 = 63\n\n")

        file.write("Train-Test Split:\n")
        file.write(f"Training samples: {len(X_train)}\n")
        file.write(f"Testing samples: {len(X_test)}\n\n")

        file.write("Model Details:\n")
        file.write("Model used: Random Forest Classifier\n")
        file.write(f"Model accuracy: {accuracy}\n\n")

        file.write("Samples Per Class:\n")
        file.write(str(sample_counts))
        file.write("\n\n")

        file.write("Classification Report:\n")
        file.write(report)

    print("\nText report saved at:")
    print(REPORT_TEXT_PATH)

    # --------------------------------------------------------
    # Step 5: Save confusion matrix image
    # --------------------------------------------------------

    labels = sorted(y.unique())

    cm = confusion_matrix(y_test, y_pred, labels=labels)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    plt.figure(figsize=(14, 14))
    display.plot(xticks_rotation=90)
    plt.title("Confusion Matrix - Sign Lingo Model")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()

    print("\nConfusion matrix saved at:")
    print(CONFUSION_MATRIX_PATH)


# ------------------------------------------------------------
# Step 6: Run main function
# ------------------------------------------------------------

if __name__ == "__main__":
    main()