"""
Feature engineering utilities using pandas and ta (technical analysis) library,
with added risk-based features using the risk_manager.
"""
import pandas as pd
import numpy as np
import ta
from src.risk.risk_manager import position_size


def add_risk_features(df: pd.DataFrame, account_equity=10000, risk_per_trade_pct=0.01, stop_distance_in_quote=0.0012, max_leverage=30):
    """
    Add risk-based features to df:
    - pos_size_units
    - pos_used_leverage
    - pos_notional
    Assumes close price as current price.
    """
    units_list = []
    leverage_list = []
    notional_list = []

    for _, row in df.iterrows():
        price = row['close']
        try:
            pos = position_size(
                account_equity=account_equity,
                risk_per_trade_pct=risk_per_trade_pct,
                stop_distance_in_quote=stop_distance_in_quote,
                price=price,
                max_leverage=max_leverage
            )
        except Exception:
            pos = {'units': 0, 'notional': 0, 'used_leverage': 0}

        units_list.append(pos['units'])
        leverage_list.append(pos['used_leverage'])
        notional_list.append(pos['notional'])

    df = df.copy()
    df['pos_size_units'] = units_list
    df['pos_used_leverage'] = leverage_list
    df['pos_notional'] = notional_list
    return df


def make_features(df: pd.DataFrame):
    df = df.copy()
    df['logret'] = np.log(df['close']).diff()

    # Technical indicators with rolling windows
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['roc'] = ta.momentum.roc(df['close'], window=10)
    df['ma_fast'] = df['close'].rolling(window=10, min_periods=1).mean()
    df['ma_slow'] = df['close'].rolling(window=50, min_periods=1).mean()
    df['ma_spread'] = df['ma_fast'] - df['ma_slow']

    df = df.dropna().reset_index(drop=True)

    # Add risk features (you can tune these params)
    df = add_risk_features(df)

    return df
