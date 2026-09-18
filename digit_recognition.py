import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import seaborn as sns


# -----------------------------------------
# 1. Load Dataset
# -----------------------------------------

digits = load_digits()

X = digits.data
y = digits.target

print("Dataset loaded successfully!")
print("Total images:", len(X))
print("Image size:", digits.images[0].shape)


# -----------------------------------------
# 2. Display Sample Images
# -----------------------------------------

plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(digits.images[i], cmap="gray")
    plt.title("Digit: " + str(y[i]))
    plt.axis("off")

plt.tight_layout()
plt.show()


# -----------------------------------------
# 3. Split Dataset
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# -----------------------------------------
# 4. Create KNN Model
# -----------------------------------------

model = KNeighborsClassifier(n_neighbors=3)


# -----------------------------------------
# 5. Train Model
# -----------------------------------------

model.fit(X_train, y_train)

print("Model training completed!")


# -----------------------------------------
# 6. Make Predictions
# -----------------------------------------

y_pred = model.predict(X_test)


# -----------------------------------------
# 7. Calculate Accuracy
# -----------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:",
      round(accuracy * 100, 2), "%")


# -----------------------------------------
# 8. Classification Report
# -----------------------------------------

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -----------------------------------------
# 9. Confusion Matrix
# -----------------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted Digit")
plt.ylabel("Actual Digit")
plt.title("Confusion Matrix")

plt.show()


# -----------------------------------------
# 10. Display Predictions
# -----------------------------------------

plt.figure(figsize=(12, 6))

for i in range(15):

    plt.subplot(3, 5, i + 1)

    plt.imshow(
        X_test[i].reshape(8, 8),
        cmap="gray"
    )

    plt.title(
        "Actual: " + str(y_test[i]) +
        "\nPredicted: " + str(y_pred[i])
    )

    plt.axis("off")

plt.tight_layout()
plt.show()
