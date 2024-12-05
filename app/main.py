from fastapi import FastAPI
from app.api import mp_hand
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()  # Create a FastAPI app instance

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(mp_hand.router, prefix="/hands", tags=["hands"])

# Define your first endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to your FastAPI app!"}
