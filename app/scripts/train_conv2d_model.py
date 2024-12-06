import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import os
import numpy as np
from sklearn.model_selection import train_test_split

current_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the folder
data_folder = os.path.join(current_directory, 'data', 'hand')

labels = []
X = np.empty((0, 30, 6, 2))
y = []
labelInt = 0

for folder_name in os.listdir(data_folder):
    labels.append(folder_name)
    
    folder_path = os.path.join(data_folder, folder_name)
    
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        example = np.load(file_path)
        X = np.append(X, example[np.newaxis,...], axis=0)
        y.append(labelInt)
        # print(example)
        
    labelInt += 1
        
        
y = np.array(y)
print(y.shape)
    

print(labels)

# Define the model
model = models.Sequential()

# Add the first 2D Convolutional layer with 32 filters, a kernel size of (3, 3), and 'relu' activation
model.add(layers.Input((30,6,2)))

model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='SAME'))

# Add a MaxPooling layer to reduce the spatial dimensions
model.add(layers.MaxPooling2D((2, 2)))

# Add a second 2D Convolutional layer with 64 filters, a kernel size of (3, 3), and 'relu' activation
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='SAME'))

# Add another MaxPooling layer
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the output from the convolutional layers
model.add(layers.Flatten())

# Add a Dense (fully connected) layer with 128 units and 'relu' activation
model.add(layers.Dense(128, activation='relu'))

# Output layer with softmax activation for classification (adjust number of units depending on classes)
model.add(layers.Dense(len(labels), activation='softmax'))  # For example, 10 classes

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summary of the model architecture
model.summary()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), verbose=2)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

model.save('scripts/conv2d_model.keras')