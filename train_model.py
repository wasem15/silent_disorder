import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

# Load the pre-saved arrays
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

print("Data loaded successfully.")

# Encode labels as integers
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Reshape X to add a channel dimension (required for CNNs)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1, 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1, 1))

# Define the CNN model
model = Sequential([
    InputLayer(input_shape=(X_train.shape[1], 1, 1)),  # Input layer with correct shape
    Conv2D(32, (3, 1), activation='relu'),             # Adjusted kernel size
    MaxPooling2D((2, 1)),                              # Pooling to reduce dimensions
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')  # Output layer with number of classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_encoded, epochs=10, validation_data=(X_test, y_test_encoded))

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)

accuracy = accuracy_score(y_test_encoded, y_pred_classes)
f1 = f1_score(y_test_encoded, y_pred_classes, average='weighted')
conf_matrix = confusion_matrix(y_test_encoded, y_pred_classes)

print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)
model.save('speech_classification_model.h5')
