import pandas as pd
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    return df

def analyze_sentiment(text: str) -> float:
    result = sentiment_pipeline(text)[0]
    score = 1.0 if result['label'] == 'POSITIVE' else -1.0
    return score
