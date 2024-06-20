from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from multi_modal_model import MultiModalStockPredictor

app = FastAPI()

class PredictionRequest(BaseModel):
    tech_features: list  
    sentiment_features: list  #

tech_input_size = 5         
sentiment_input_size = 1    
hidden_size = 50
num_layers = 2
output_size = 1

model = MultiModalStockPredictor(tech_input_size, sentiment_input_size, hidden_size, num_layers, output_size)
model.load_state_dict(torch.load("multi_modal_stock_predictor.pth", map_location=torch.device('cpu')))
model.eval()

@app.post("/predict")
def predict_stock(data: PredictionRequest):
    try:
        tech_features = np.array(data.tech_features, dtype=np.float32)
        sentiment_features = np.array(data.sentiment_features, dtype=np.float32)
        tech_features_tensor = torch.tensor(tech_features).unsqueeze(0)
        sentiment_features_tensor = torch.tensor(sentiment_features).unsqueeze(0)
        with torch.no_grad():
            prediction = model(tech_features_tensor, sentiment_features_tensor)
        return {"prediction": prediction.item()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
