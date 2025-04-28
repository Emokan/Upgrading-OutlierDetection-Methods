# https://www.kaggle.com/code/kanncaa1/convolutional-neural-network-cnn-tutorial

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import warnings
warnings.filterwarnings('ignore')
from keras.api.utils import to_categorical
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.api.layers import RandomRotation, RandomZoom, RandomTranslation
from keras.api.optimizers import Adam

# Veri Seti
digits = load_digits()
X = digits.images
Y = digits.target

print("X shape:", X.shape)  # (1797, 8, 8)
print("Y shape:", Y.shape)  # (1797,)

plt.imshow(X[0], cmap='gray')
plt.title(f"Label: {Y[0]}")
plt.show()

# Normalize
X = X / 16.0
X = X.reshape(-1, 8, 8, 1)

# Label Encode
Y = to_categorical(Y, num_classes=10)

# Train-Test Split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=2)

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("Y_train shape:", Y_train.shape)
print("Y_val shape:", Y_val.shape)

# Model
model = Sequential()

# Data Augmentation Katmanları
model.add(RandomRotation(0.05))          # %5 rotation
model.add(RandomZoom(0.1))                # %10 zoom
model.add(RandomTranslation(0.1, 0.1))    # %10 width ve height shift

# Conv2D Katmanları
model.add(Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu', input_shape=(8,8,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

# Compile
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Eğitim
epochs = 15
batch_size = 32

history = model.fit(X_train, Y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_val, Y_val))

# Loss Grafiği
plt.plot(history.history['val_loss'], color='b', label="Validation Loss")
plt.plot(history.history['loss'], color='r', label="Training Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Prediction ve Confusion Matrix
Y_pred = model.predict(X_val)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_val, axis=1)

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

plt.figure(figsize=(8,8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01, cmap="Greens", linecolor="gray", fmt='.1f')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()