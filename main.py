import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

from data_ingestion import StockDataLoader, NewsDataLoader
from feature_engineering import compute_technical_indicators, aggregate_daily_sentiment
from multi_modal_model import MultiModalStockPredictor
from train import train_model

def create_sequences(tech_data: np.ndarray, sentiment_data: np.ndarray, seq_length: int):
    tech_seqs, sentiment_seqs, ys = [], [], []
    for i in range(len(tech_data) - seq_length):
        tech_seq = tech_data[i:i+seq_length]
        sentiment_seq = sentiment_data[i:i+seq_length]
        target = tech_data[i+seq_length, 0]  
        tech_seqs.append(tech_seq)
        sentiment_seqs.append(sentiment_seq)
        ys.append(target)
    return np.array(tech_seqs), np.array(sentiment_seqs), np.array(ys)

def main():
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    seq_length = 20
    num_epochs = 50
    batch_size = 32
    learning_rate = 0.001

    stock_loader = StockDataLoader(ticker, start_date, end_date)
    stock_df = stock_loader.fetch_data()
    stock_df['Date'] = stock_df['Date'].dt.date

    news_api_key = "NEWSAPI_KEY"  
    news_loader = NewsDataLoader(news_api_key, ticker, start_date, end_date)
    news_df = news_loader.fetch_news()

    stock_df = compute_technical_indicators(stock_df)
    stock_df.dropna(inplace=True)

    sentiment_df = aggregate_daily_sentiment(news_df)
    merged_df = pd.merge(stock_df, sentiment_df, left_on='Date', right_on='Date', how='left')
    merged_df['Daily_Sentiment'].fillna(0, inplace=True)

    tech_features = merged_df[['Close', 'SMA_20', 'EMA_20', 'RSI', 'MACD']].values
    sentiment_features = merged_df[['Daily_Sentiment']].values

    scaler_tech = MinMaxScaler()
    tech_features_scaled = scaler_tech.fit_transform(tech_features)
    scaler_sent = MinMaxScaler()
    sentiment_features_scaled = scaler_sent.fit_transform(sentiment_features)

    X_tech, X_sentiment, y = create_sequences(tech_features_scaled, sentiment_features_scaled, seq_length)
    X_tech_tensor = torch.tensor(X_tech, dtype=torch.float32)
    X_sent_tensor = torch.tensor(X_sentiment, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_tech_tensor, X_sent_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    tech_input_size = X_tech_tensor.shape[2]       
    sentiment_input_size = X_sent_tensor.shape[2]  
    hidden_size = 50
    num_layers = 2
    output_size = 1

    model = MultiModalStockPredictor(tech_input_size, sentiment_input_size, hidden_size, num_layers, output_size)

    train_model(model, train_loader, num_epochs=num_epochs, learning_rate=learning_rate, multi_modal=True)

    torch.save(model.state_dict(), "multi_modal_stock_predictor.pth")
    print("Model saved as multi_modal_stock_predictor.pth")

if __name__ == "__main__":
    main()
