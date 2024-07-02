import torch
import numpy as np
from multi_modal_model import MultiModalStockPredictor

def predict_with_uncertainty(model, tech_input, sentiment_input, num_samples=50):
    """
    Monte Carlo Dropout to estimate prediction uncertainty
    
    Args:
        model: The trained model that has dropout layers.
        tech_input: Tensor for technical features with shape [1, seq_len, tech_input_size].
        sentiment_input: Tensor for sentiment features with shape [1, seq_len, sentiment_input_size].
        num_samples: Number of stochastic forward passes.
    
    Returns:
        A tuple (mean_prediction, std_prediction) where:
          - mean_prediction is the average prediction over num_samples.
          - std_prediction is the standard deviation (uncertainty measure).
    """
    model.train()  
    predictions = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            pred = model(tech_input, sentiment_input)
            predictions.append(pred.item())
    
    predictions = np.array(predictions)
    mean_pred = predictions.mean()
    std_pred = predictions.std()
    return mean_pred, std_pred


