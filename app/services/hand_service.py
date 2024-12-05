import os
import numpy as np
from datetime import datetime
from app.schemas.hand import HandData

def store_hand_data(hand_data: HandData):
    path = f"app/data/hand/{hand_data.label}/"
    os.makedirs(path, exist_ok=True) 
    
    hand_landmarks_array = np.array(hand_data.landmarks)
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = os.path.join(path,f"{timestamp}.npy")
    
    np.save(filename, hand_landmarks_array)
    
    return {"status": "success"}