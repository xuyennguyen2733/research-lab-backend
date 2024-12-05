from pydantic import BaseModel
import numpy as np
from typing import List

class HandData(BaseModel):
    label: str
    landmarks: List[List[float]] # [frames][fingers][x and y coordinates]
    