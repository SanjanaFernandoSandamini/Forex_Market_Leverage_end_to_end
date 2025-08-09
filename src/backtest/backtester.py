import pandas as pd
import numpy as np

def backtest_signals(df, signals, initial_cash=10000, fee_per_trade=0.0, spread=0.0001, leverage=1.0):
    cash = initial_cash
    position = 0.0
    equity_curve = []
    position_entry_price = 0.0
    
    for idx, row in df.iterrows():
        price = row['close']
        sig = signals.iloc[idx]
        
        # Enter position
        if sig == 1 and position == 0:
            position = leverage  # number of units is scaled by leverage
            position_entry_price = price
            cost = price * position * (1 + spread) + fee_per_trade
            cash -= cost
        
        # Exit position
        elif sig == 0 and position > 0:
            proceeds = price * position * (1 - spread) - fee_per_trade
            cash += proceeds
            position = 0.0
        
        # Mark to market equity calculation
        mark = cash + position * price
        equity_curve.append(mark)
    
    return pd.Series(equity_curve, index=df.index)
