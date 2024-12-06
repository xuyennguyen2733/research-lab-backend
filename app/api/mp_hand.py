from fastapi import APIRouter, HTTPException
from app.services.hand_service import store_hand_data, predict_hand_shape
from app.schemas.hand import HandData

router = APIRouter()

@router.post("/", response_model=dict)
async def collect_hand_data(hand_data: HandData):
    try:
        return store_hand_data(hand_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling api: {str(e)}")
    
@router.post("/predict", response_model=dict)
async def predict_hand(hand_data: HandData):
    try: 
        return predict_hand_shape(hand_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to predict: {str(e)}")
