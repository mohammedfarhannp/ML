# Import Modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from sklearn.datasets import load_digits

import numpy as np

# Load Dataset
digits = load_digits()

x = digits.images
y = digits.target.reshape(-1, 1)

# One-Hot Encode Targets
Encoder = OneHotEncoder(sparse_output=False)
y = Encoder.fit_transform(y)

# Reshape for CNN
x = x.reshape(-1, 8, 8, 1)

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Model Creation
Model = Sequential()

Model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(8, 8, 1)))
Model.add(MaxPooling2D((2, 2)))

Model.add(Conv2D(64, (3, 3), activation='relu'))   # removed 2nd pooling!

Model.add(Flatten())
Model.add(Dense(64, activation='relu'))
Model.add(Dense(10, activation='softmax'))

Model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
Model.fit(x_train, y_train, epochs=10, batch_size=32)

# Test Prediction
Loss, Acc = Model.evaluate(x_test, y_test)
print("\nTest Accuracy :", Acc)

# New Data Prediction
new_data = x_test[0].reshape(1, 8, 8, 1)

prediction = Model.predict(new_data)[0]
predicted_class = np.argmax(prediction)

print("\n===== New Prediction =====")
print("Predicted Digit :", predicted_class)
print("Probabilities :", prediction)
