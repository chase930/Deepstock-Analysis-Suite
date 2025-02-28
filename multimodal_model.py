import torch
import torch.nn as nn

class MultiModalStockPredictor(nn.Module):
    def __init__(self, tech_input_size, sentiment_input_size, hidden_size, num_layers, output_size, dropout=0.5):
        super(MultiModalStockPredictor, self).__init__()
        self.tech_lstm = nn.LSTM(tech_input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.sentiment_lstm = nn.LSTM(sentiment_input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, tech_x, sentiment_x):
        tech_out, _ = self.tech_lstm(tech_x)
        sentiment_out, _ = self.sentiment_lstm(sentiment_x)
        tech_last = tech_out[:, -1, :]
        sentiment_last = sentiment_out[:, -1, :]
        fused = torch.cat((tech_last, sentiment_last), dim=1)
        fused = self.dropout(fused)  
        output = self.fc(fused)
        return output
