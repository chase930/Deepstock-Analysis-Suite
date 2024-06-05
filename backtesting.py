import pandas as pd
import numpy as np

def simple_backtest(predictions, actual_prices, initial_capital=10000):
    """
    - If the predicted next-day price > current price, buy (if not already holding)
    - If predicted next-day price < current price, sell (if holding)
    """
    capital = initial_capital
    position = 0
    portfolio_values = []
    
    for i in range(len(predictions) - 1):
        current_price = actual_prices[i]
        if predictions[i] > current_price and position == 0:
            position = capital / current_price  
            capital = 0
        elif predictions[i] < current_price and position > 0:
            capital = position * current_price
            position = 0
        portfolio_value = capital + position * current_price
        portfolio_values.append(portfolio_value)
    
    final_value = capital + position * actual_prices[-1]
    returns = (final_value - initial_capital) / initial_capital * 100
    return portfolio_values, final_value, returns

if __name__ == "__main__":
    dates = pd.date_range(start="2023-01-01", periods=100)
    actual_prices = np.linspace(100, 150, 100)
    predictions = actual_prices + np.random.normal(0, 2, 100)
    portfolio_values, final_value, returns = simple_backtest(predictions, actual_prices)
    print(f"Final portfolio value: ${final_value:.2f}, Returns: {returns:.2f}%")
