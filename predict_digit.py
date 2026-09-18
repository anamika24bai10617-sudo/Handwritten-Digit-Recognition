import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# Load dataset
digits = load_digits()

X = digits.data
y = digits.target


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create and train model
model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, y_train)


# Select an image for prediction
index = 10

image = X_test[index].reshape(8, 8)

prediction = model.predict(
    X_test[index].reshape(1, -1)
)[0]


# Display result
plt.figure(figsize=(4, 4))

plt.imshow(image, cmap="gray")

plt.title(
    "Predicted Digit: " + str(prediction) +
    "\nActual Digit: " + str(y_test[index])
)

plt.axis("off")

plt.show()

print("Predicted Digit:", prediction)
print("Actual Digit:", y_test[index])
