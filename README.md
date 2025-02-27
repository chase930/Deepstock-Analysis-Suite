# DeepStock Analysis Suite

## Overview
DeepStock Analysis Suite is a modular Python project built on PyTorch that performs advanced stock analysis by fusing technical indicators with news sentiment data. The suite is designed to:
- Fetch and preprocess historical stock data and financial news.
- Compute technical indicators (SMA, EMA, RSI, MACD) and aggregate daily news sentiment.
- Build and train a multi-modal deep learning model with separate LSTM branches for technical and sentiment inputs.
- Backtest trading strategies based on model predictions.
- Estimate prediction uncertainty using Monte Carlo Dropout.
- Deploy predictions via a FastAPI endpoint for real-time use.

## How I Solved the Problem
The project addresses the challenge of forecasting stock prices by integrating diverse data sources and applying advanced deep learning techniques:
- **Data Collection:**  
  - *Stock Data:* Collected via `yfinance` from Yahoo Finance.
  - *News Data:* Retrieved using NewsAPI, then processed with a transformer-based sentiment analysis pipeline.
- **Feature Engineering:**  
  - Computed technical indicators to capture market trends.
  - Performed sentiment analysis on news headlines and descriptions, aggregating scores on a daily basis.
- **Model Building:**  
  - Designed a multi-modal model (`MultiModalStockPredictor`) in PyTorch that processes both technical and sentiment data through separate LSTM branches before fusing them.
  - Integrated dropout layers to enable Monte Carlo Dropout for uncertainty estimation.
- **Training & Backtesting:**  
  - Developed a training pipeline that cleans, sequences, and feeds data into the model.
  - Implemented a robust backtesting module that simulates trading decisions, tracks portfolio performance, and computes key performance metrics.
- **Deployment & Uncertainty Estimation:**  
  - Built a FastAPI-based deployment module to serve real-time predictions.
  - Added an uncertainty estimation module that leverages dropout during inference to provide confidence intervals.

## Function Overview
- **Data Ingestion (`data_ingestion.py`):**
  - **StockDataLoader:** Downloads historical stock prices from Yahoo Finance.
  - **NewsDataLoader:** Retrieves financial news articles using NewsAPI.
  
- **Feature Engineering (`feature_engineering.py`):**
  - **compute_technical_indicators:** Calculates SMA, EMA, RSI, and MACD from historical stock data.
  - **analyze_sentiment:** Applies a sentiment analysis pipeline to convert news text into numerical sentiment scores.
  - **aggregate_daily_sentiment:** Averages sentiment scores on a daily basis.
  
- **Model Building (`multi_modal_model.py`):**
  - **MultiModalStockPredictor:** A PyTorch model that includes two LSTM branches—one for technical features and one for sentiment features. Outputs are fused via a fully connected layer. Dropout layers are included to facilitate uncertainty estimation.
  
- **Training (`train.py` & `main.py`):**
  - Contains a training loop using PyTorch, sequence generation from preprocessed data, and saving of trained models.
  
- **Backtesting (`backtesting.py`):**
  - Simulates trading strategies based on model predictions.
  - Tracks portfolio performance over time.
  - Calculates metrics such as total return, maximum drawdown, and Sharpe ratio.
  
- **Deployment (`api.py`):**
  - Provides a FastAPI endpoint to serve real-time stock predictions.
  
- **Uncertainty Estimation (`uncertainty.py`):**
  - Uses Monte Carlo Dropout by running multiple stochastic forward passes through the model.
  - Returns both the mean prediction and a standard deviation as an uncertainty measure.

