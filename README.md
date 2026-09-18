# Handwritten Digit Recognition Using KNN

## 📌 Project Overview

This project implements a Computer Vision system for recognizing
handwritten digits using the K-Nearest Neighbors (KNN) machine
learning algorithm.

The system recognizes digits from 0 to 9.

---

## 🎯 Objective

The main objective of this project is to develop a simple
Computer Vision system that can identify handwritten digits
from images.

---

## 📊 Dataset

The project uses the Digits Dataset provided by Scikit-learn.

Dataset details:

- Total images: 1797
- Number of classes: 10
- Classes: 0 to 9
- Image size: 8 × 8 pixels

---

## 🛠️ Technologies Used

- Python
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn
- Google Colab
- GitHub

---

## 🧠 Algorithm

The project uses the K-Nearest Neighbors (KNN) algorithm.

KNN classifies an image based on the nearest training examples
in the feature space.

---

## 🔄 Methodology

The project follows these steps:

1. Load the handwritten digit dataset
2. Visualize sample images
3. Split the dataset into training and testing data
4. Create the KNN classifier
5. Train the model
6. Generate predictions
7. Calculate accuracy
8. Generate classification report
9. Generate confusion matrix
10. Test individual digit images

---

## 📈 Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 📁 Project Structure

```text
Handwritten-Digit-Recognition/
│
├── README.md
├── requirements.txt
├── digit_recognition.py
└── predict_digit.py
