import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

class DummyRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = DummyRegressor()
model.load_state_dict(torch.load("dummy_model.pt", map_location=torch.device("cpu")))
model.eval()

class PredictRequest(BaseModel):
    features: List[List[float]]  # Each inner list must have 1 float

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        X = torch.tensor(request.features, dtype=torch.float32)
        with torch.no_grad():
            preds = model(X).squeeze().tolist()
        return {"predictions": preds}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))