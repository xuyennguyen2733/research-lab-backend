from fastapi import APIRouter, HTTPException
from app.services.hand_service import store_hand_data
from app.schemas.hand import HandData

router = APIRouter()

@router.post("/", response_model=dict)
async def collect_hand_data(hand_data: HandData):
    try:
        return store_hand_data(hand_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling api: {str(e)}")