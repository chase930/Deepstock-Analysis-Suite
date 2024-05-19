import yfinance as yf
import pandas as pd
import requests

class StockDataLoader:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self):
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        data.reset_index(inplace=True)
        return data

class NewsDataLoader:
    def __init__(self, api_key, query, start_date, end_date):
        self.api_key = api_key
        self.query = query
        self.start_date = start_date
        self.end_date = end_date

    def fetch_news(self):
        url = (
            'https://newsapi.org/v2/everything?'
            f'q={self.query}&'
            f'from={self.start_date}&'
            f'to={self.end_date}&'
            'sortBy=popularity&'
            'language=en&'
            f'apiKey={self.api_key}'
        )
        response = requests.get(url)
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            news_df = pd.DataFrame(articles)
            if not news_df.empty and 'publishedAt' in news_df.columns:
                news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt']).dt.date
            return news_df
        else:
            print("Error fetching news:", response.status_code)
            return pd.DataFrame()
