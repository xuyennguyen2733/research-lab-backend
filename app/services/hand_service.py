import os
import numpy as np
from datetime import datetime
from app.schemas.hand import HandData
from keras import models

def store_hand_data(hand_data: HandData):
    path = f"app/data/hand/{hand_data.label}/"
    os.makedirs(path, exist_ok=True) 
    
    hand_landmarks_array = np.array(hand_data.landmarks)
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = os.path.join(path,f"{timestamp}_{hand_data.index}.npy")
    
    np.save(filename, hand_landmarks_array)
    
    return {"status": "success"}

def predict_hand_shape(hand_data: HandData):
    path = f"app/models/conv2d_model.keras"
    model = models.load_model(path)
    labels = ['a', 'j', 'z']
    landmarks = np.array(hand_data.landmarks)
    landmarks = landmarks[np.newaxis, ...]
    predictions = model.predict(landmarks)
    
    print('predictions', predictions)
    
    predicted_class = np.argmax(predictions[0])
    predicted_sign = labels[predicted_class]
    
    return {"status": predicted_sign}
    
    