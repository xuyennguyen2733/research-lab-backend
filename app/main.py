from fastapi import FastAPI
from app.api import mp_hand

app = FastAPI()  # Create a FastAPI app instance

app.include_router(mp_hand.router, prefix="/hands", tags=["hands"])

# Define your first endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to your FastAPI app!"}
