import tensorflow as tf
from keras import layers, models
import os
import numpy as np
from keras import models

current_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the folder
data_folder = os.path.join(current_directory, 'test_data', 'hand')
model_path = os.path.join(current_directory, 'models', "conv2d_model.keras")

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
model = models.load_model(model_path)
test_loss, test_acc = model.evaluate(X, y, verbose=2)
print(f"Test accuracy: {test_acc}")