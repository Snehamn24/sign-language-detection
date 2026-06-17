# train_model.py
# ------------------------------------------------------------
# This file trains a simple Machine Learning model using the
# hand landmark CSV file created by MediaPipe.
#
# Input:
# dataset/landmarks.csv
#
# Output:
# models/sign_lingo_model.pkl
#
# We are NOT using CNN here.
# We are using landmark values + Random Forest classifier.
# ------------------------------------------------------------

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


# ------------------------------------------------------------
# Step 1: Define project paths
# ------------------------------------------------------------

# ROOT_DIR points to the main project folder
ROOT_DIR = Path(__file__).resolve().parents[1]

# CSV file created from extract_landmarks_from_dataset.py
CSV_PATH = ROOT_DIR / "dataset" / "landmarks.csv"

# Folder where trained model will be saved
MODEL_DIR = ROOT_DIR / "models"

# Final trained model path
MODEL_PATH = MODEL_DIR / "sign_lingo_model.pkl"


# ------------------------------------------------------------
# Step 2: Load dataset
# ------------------------------------------------------------

def main():
    # Check whether landmarks.csv exists
    if not CSV_PATH.exists():
        print("landmarks.csv not found!")
        print("Expected path:", CSV_PATH)
        return

    # Read CSV file using pandas
    data = pd.read_csv(CSV_PATH)

    print("Dataset loaded successfully.")
    print("Dataset shape:", data.shape)

    # Show number of samples for each class
    print("\nSamples per class:")
    print(data["label"].value_counts())


    # --------------------------------------------------------
    # Step 3: Split input features and target labels
    # --------------------------------------------------------

    # X contains landmark values: x0,y0,z0 ... x20,y20,z20
    X = data.drop("label", axis=1)

    # y contains the output class: A, B, C, ..., space, del, nothing
    y = data["label"]


    # --------------------------------------------------------
    # Step 4: Split data into training and testing parts
    # --------------------------------------------------------

    # 80% data is used for training
    # 20% data is used for testing
    #
    # stratify=y makes sure every class is properly represented
    # in both training and testing data.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])


    # --------------------------------------------------------
    # Step 5: Create the machine learning model
    # --------------------------------------------------------

    # RandomForestClassifier is used because:
    # 1. It works well for landmark/numerical data
    # 2. It is easier than CNN
    # 3. It trains faster
    # 4. It is easy to explain in presentation
    model = RandomForestClassifier(
        n_estimators=200,      # Number of decision trees
        random_state=42,
        n_jobs=-1              # Use all CPU cores
    )


    # --------------------------------------------------------
    # Step 6: Train the model
    # --------------------------------------------------------

    print("\nTraining model...")
    model.fit(X_train, y_train)

    print("Training completed.")


    # --------------------------------------------------------
    # Step 7: Test the model
    # --------------------------------------------------------

    # Predict labels for test data
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("\nModel Accuracy:", accuracy)

    # Print detailed class-wise performance
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


    # --------------------------------------------------------
    # Step 8: Save the trained model
    # --------------------------------------------------------

    # Create models folder if it does not exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save model using joblib
    joblib.dump(model, MODEL_PATH)

    print("\nModel saved successfully at:")
    print(MODEL_PATH)


# ------------------------------------------------------------
# Step 9: Run main function
# ------------------------------------------------------------

if __name__ == "__main__":
    main()