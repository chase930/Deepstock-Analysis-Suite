import pandas as pd
import numpy as np
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['RSI'] = 100 - (100 / (1 + rs))
    
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26

    return df

def analyze_sentiment(text: str) -> float:
    result = sentiment_pipeline(text)[0]
    score = 1.0 if result['label'] == 'POSITIVE' else -1.0
    return score

def aggregate_daily_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    if news_df.empty:
        return pd.DataFrame()
    news_df['combined_text'] = news_df['title'].fillna('') + ' ' + news_df['description'].fillna('')
    news_df['sentiment_score'] = news_df['combined_text'].apply(analyze_sentiment)
    daily_sentiment = news_df.groupby('publishedAt')['sentiment_score'].mean().reset_index()
    daily_sentiment.rename(columns={'publishedAt': 'Date', 'sentiment_score': 'Daily_Sentiment'}, inplace=True)
    return daily_sentiment
