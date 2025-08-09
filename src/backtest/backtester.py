"""
Very simple backtester skeleton. Use this to simulate entry/exit given signals.
"""
import pandas as pd
import numpy as np


def backtest_signals(df, signals, initial_cash=10000, fee_per_trade=0.0, spread=0.0001):
    cash = initial_cash
    position = 0.0
    equity_curve = []
    for idx, row in df.iterrows():
        price = row['close']
        sig = signals.iloc[idx]
        # very simple: sig=1 -> long 1 unit, sig=0 -> flat
        if sig == 1 and position == 0:
            position = 1.0
            cash -= price * position * (1 + spread) + fee_per_trade
        elif sig == 0 and position == 1:
            cash += price * position * (1 - spread) - fee_per_trade
            position = 0
        mark = cash + position * price
        equity_curve.append(mark)
    return pd.Series(equity_curve, index=df.index)
