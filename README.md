# Sign Lingo: Real-Time Sign Language Alphabet Recognition and Sentence Formation

Sign Lingo is a real-time sign language recognition project that detects ASL alphabet hand signs and converts them into text. The system recognizes alphabet signs using webcam input and appends the detected letters to form words and sentences.

This project uses a lightweight landmark-based machine learning approach instead of a heavy CNN model. MediaPipe is used to extract hand landmarks, and a Random Forest classifier is used to recognize signs.

---

## Project Objective

The main objective of this project is to help convert hand signs into readable text in real time.

The system performs the following tasks:

* Detects hand signs using webcam
* Extracts 21 hand landmarks using MediaPipe
* Predicts ASL alphabet signs
* Appends predicted letters to form words
* Supports sentence formation using `space`
* Supports deletion using `del`

---

## Technologies Used

* Python
* OpenCV
* MediaPipe
* Scikit-learn
* Pandas
* NumPy
* Joblib
* Matplotlib

---

## Dataset Used

Dataset: **ASL Alphabet Dataset**

Source: Kaggle
Link: https://www.kaggle.com/datasets/grassknoted/asl-alphabet

The dataset contains images of ASL alphabet signs. It includes the following classes:

* A to Z alphabets
* `space`
* `del`
* `nothing`

In this project, the image dataset is not directly trained using CNN. Instead, MediaPipe is used to extract hand landmark points from the images, and those landmark values are used to train a machine learning model.

---

## Project Workflow

```text
ASL Alphabet Image Dataset
        ↓
MediaPipe Hand Landmark Extraction
        ↓
landmarks.csv file creation
        ↓
Random Forest model training
        ↓
Real-time webcam prediction
        ↓
Sentence formation
```

---

## Model Used

The trained model used in this project is:

```text
Random Forest Classifier
```

MediaPipe detects 21 hand landmarks. Each landmark contains:

```text
x, y, z coordinates
```

So each hand image is converted into:

```text
21 × 3 = 63 features
```

These 63 features are passed to the Random Forest classifier for sign prediction.

---

## Why This Approach Was Chosen

A CNN model requires full image-based training, which can take more time and computational resources.

In this project, a landmark-based approach was selected because:

* It is faster to train
* It works well for real-time prediction
* It is lightweight
* It is easier to explain and demonstrate
* It gives good accuracy using hand landmark features

---

## Project Folder Structure

```text
sign-language/
│
├── dataset/
│   ├── asl_alphabet_train/
│   │   ├── A/
│   │   ├── B/
│   │   ├── C/
│   │   ├── ...
│   │   ├── Z/
│   │   ├── del/
│   │   ├── nothing/
│   │   └── space/
│   │
│   ├── asl_alphabet_test/
│   └── landmarks.csv
│
├── models/
│   └── sign_lingo_model.pkl
│
├── reports/
│   ├── project_statistics.txt
│   └── confusion_matrix.png
│
├── src/
│   ├── labels.py
│   ├── extract_landmarks_from_dataset.py
│   ├── train_model.py
│   ├── predict_sentence.py
│   └── project_statistics.py
│
├── screenshots/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation and Setup

### 1. Clone the repository

```bash
git clone <your-github-repository-link>
cd sign-language
```

### 2. Create virtual environment

```bash
python -m venv venv
```

### 3. Activate virtual environment

For Windows CMD:

```bash
venv\Scripts\activate
```

For Windows PowerShell:

```bash
.\venv\Scripts\activate
```

### 4. Install required packages

```bash
pip install -r requirements.txt
```

---

## Requirements

The `requirements.txt` file contains:

```text
opencv-python
mediapipe==0.10.14
pandas
numpy
scikit-learn
joblib
matplotlib
```

---

## Dataset Setup

Download the ASL Alphabet Dataset from Kaggle:

```text
https://www.kaggle.com/datasets/grassknoted/asl-alphabet
```

After downloading and unzipping, place the dataset inside the `dataset` folder.

The final dataset structure should be:

```text
dataset/
└── asl_alphabet_train/
    ├── A/
    ├── B/
    ├── C/
    ├── ...
    ├── Z/
    ├── del/
    ├── nothing/
    └── space/
```

The class folders should be directly inside `asl_alphabet_train`.

Correct path example:

```text
dataset/asl_alphabet_train/A
```

---

## Steps to Run the Project

### Step 1: Extract Hand Landmarks

Run:

```bash
py src/extract_landmarks_from_dataset.py
```

This script reads the ASL image dataset, extracts hand landmarks using MediaPipe, and creates:

```text
dataset/landmarks.csv
```

---

### Step 2: Train the Model

Run:

```bash
py src/train_model.py
```

This trains the Random Forest classifier and saves the model as:

```text
models/sign_lingo_model.pkl
```

---

### Step 3: Run Real-Time Prediction

Run:

```bash
py src/predict_sentence.py
```

This opens the webcam and starts real-time sign recognition.

---

## Real-Time Controls

While running the webcam application:

| Key         | Action                          |
| ----------- | ------------------------------- |
| `q`         | Quit the application            |
| `c`         | Clear the sentence              |
| `s`         | Manually add current prediction |
| `space key` | Add keyboard space              |
| `backspace` | Delete last character           |

---

## Sentence Formation Logic

The system recognizes signs and updates the sentence as follows:

| Prediction | Action                     |
| ---------- | -------------------------- |
| A-Z        | Adds the detected letter   |
| space      | Adds a blank space         |
| del        | Deletes the last character |
| nothing    | Does nothing               |

Example:

```text
C → A → T
```

Output:

```text
CAT
```

Example:

```text
H → I → space → T → H → E → R → E
```

Output:

```text
HI THERE
```

---

## Project Statistics

The project achieved the following results:

```text
Total landmark samples saved: 13,002
Total skipped images: 1,498
Testing samples: 2,601
Model accuracy: 99%
```

Each sample contains:

```text
63 features
```

because:

```text
21 hand landmarks × 3 coordinates = 63 features
```

To generate project statistics, run:

```bash
py src/project_statistics.py
```

This creates:

```text
reports/project_statistics.txt
reports/confusion_matrix.png
```

---

## Accuracy

The Random Forest classifier achieved approximately:

```text
99% accuracy
```

on the test data.

The model showed strong performance because MediaPipe provides clean landmark features, making classification easier than using raw image pixels.

---

## Advantages of the System

* Real-time prediction
* Lightweight model
* Easy to train
* Uses publicly available dataset
* Supports word and sentence formation
* Does not require GPU
* Suitable for demonstration and academic presentation

---

## Limitations

* The system currently recognizes static alphabet signs only.
* Dynamic signs or full-word gestures are not included in the current version.
* Some signs may be affected by poor lighting or incorrect hand position.
* Repeated letters may require a small pause or manual add using the `s` key.

---

## Future Enhancements

In the future, this project can be improved by:

* Adding full-word sign recognition
* Training custom signs for common words
* Adding speech output for the generated sentence
* Improving repeated-letter handling
* Creating a GUI-based application
* Using deep learning models such as MLP or CNN-LSTM for advanced recognition

---

## Conclusion

Sign Lingo is a real-time sign language alphabet recognition and sentence formation system. It uses MediaPipe to extract hand landmarks and a Random Forest classifier to recognize ASL signs. The detected letters are appended to form words and sentences, making the system useful for basic sign-to-text communication.

This project demonstrates how computer vision and machine learning can be combined to build an assistive communication tool.
