"""
Preprocessing utilities: load raw CSVs and produce features + target for training.
"""
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import ta  # technical analysis indicators


def load_pair_csv(symbol: str, interval: str = '60min'):
    p = Path(f'data/raw/{symbol}_{interval}.csv')
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def add_returns(df: pd.DataFrame, periods: int = 1):
    df = df.copy()
    df['return'] = df['close'].pct_change(periods)
    return df


def add_technical_features(df: pd.DataFrame):
    df = df.copy()
    # RSI, ATR, ROC, Moving Average Spread
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['roc'] = ta.momentum.roc(df['close'], window=1)
    df['ma_fast'] = df['close'].rolling(window=5).mean()
    df['ma_slow'] = df['close'].rolling(window=20).mean()
    df['ma_spread'] = df['ma_fast'] - df['ma_slow']

    # Fill NaNs created by rolling calculations
    df.fillna(method='bfill', inplace=True)
    return df


def process_all(interval: str = '60min'):
    files = glob.glob('data/raw/*.csv')
    for f in files:
        df = pd.read_csv(f, parse_dates=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = add_returns(df)
        df = add_technical_features(df)
        out = f.replace('raw', 'processed')
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print('Processed', out)


if __name__ == '__main__':
    process_all()

