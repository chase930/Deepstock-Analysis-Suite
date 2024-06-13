import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, prices, predictions, initial_capital=10000, commission=0.001, slippage=0.0):
        """
        prices: array-like of actual closing prices.
        predictions: array-like of predicted next-day prices.
        initial_capital: starting cash for backtesting.
        commission: commission rate per trade (as a fraction of transaction value).
        slippage: slippage rate (as a fraction of transaction value).
        """
        self.prices = np.array(prices)
        self.predictions = np.array(predictions)
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        self.position = 0  
        self.cash = initial_capital
        self.trades = [] 
        self.portfolio_values = []  

    def run_backtest(self):
        """
        Run the backtest over the provided price and prediction series.
        Strategy:
          - If tomorrow's predicted price is higher than today's price and no position is held, buy.
          - If the prediction is lower than today's price and a position is held, sell.
        """
        for i in range(len(self.prices) - 1):
            current_price = self.prices[i]
            next_prediction = self.predictions[i]
            
            if next_prediction > current_price and self.position == 0:
                buy_price = current_price * (1 + self.slippage)
                num_shares = self.cash / buy_price
                cost = num_shares * buy_price
                commission_cost = cost * self.commission
                self.cash -= commission_cost  # subtract commission
                self.position = num_shares
                trade_entry = {
                    "entry_index": i,
                    "entry_price": buy_price,
                    "entry_commission": commission_cost,
                    "num_shares": num_shares
                }
                self.trades.append(trade_entry)

            elif next_prediction < current_price and self.position > 0:
                sell_price = current_price * (1 - self.slippage)
                proceeds = self.position * sell_price
                commission_cost = proceeds * self.commission
                proceeds -= commission_cost  
                self.cash += proceeds
                trade_exit = {
                    "exit_index": i,
                    "exit_price": sell_price,
                    "exit_commission": commission_cost,
                    "trade_return": (sell_price - self.trades[-1]["entry_price"]) / self.trades[-1]["entry_price"]
                }
                self.trades[-1].update(trade_exit)
                self.position = 0

            portfolio_value = self.cash + self.position * current_price
            self.portfolio_values.append(portfolio_value)

        final_price = self.prices[-1]
        final_portfolio_value = self.cash + self.position * final_price
        self.portfolio_values.append(final_portfolio_value)
        return self.portfolio_values

    def calculate_performance(self):
        """
        Calculate key performance metrics:
          - Total return (%)
          - Maximum drawdown (%)
          - Annualized Sharpe ratio (assuming 252 trading days)
        """
        portfolio_series = pd.Series(self.portfolio_values)
        total_return = (portfolio_series.iloc[-1] - self.initial_capital) / self.initial_capital * 100

        roll_max = portfolio_series.cummax()
        drawdown = (portfolio_series - roll_max) / roll_max
        max_drawdown = drawdown.min() * 100

        daily_returns = portfolio_series.pct_change().fillna(0)
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / (daily_returns.std() + 1e-9)

        performance = {
            "final_portfolio_value": portfolio_series.iloc[-1],
            "total_return_%": total_return,
            "max_drawdown_%": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "trades": self.trades,
            "portfolio_series": portfolio_series
        }
        return performance

# Example usage:
if __name__ == "__main__":
    dates = pd.date_range("2023-01-01", periods=100)
    prices = np.linspace(100, 150, 100) + np.random.normal(0, 2, 100)
    predictions = prices + np.random.normal(0, 2, 100)
    
    bt = Backtester(prices, predictions, initial_capital=10000, commission=0.001, slippage=0.001)
    bt.run_backtest()
    performance = bt.calculate_performance()

    print("Final portfolio value: ${:.2f}".format(performance["final_portfolio_value"]))
    print("Total return: {:.2f}%".format(performance["total_return_%"]))
    print("Max drawdown: {:.2f}%".format(performance["max_drawdown_%"]))
    print("Sharpe Ratio: {:.2f}".format(performance["sharpe_ratio"]))
    print("Trade details:")
    for trade in performance["trades"]:
        print(trade)
