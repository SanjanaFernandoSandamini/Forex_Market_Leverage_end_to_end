"""
Training entrypoint. Example: train an XGBoost classifier predicting next-bar sign of return,
including risk-based features in training data.
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from models.xgb_model import train_xgb


def prepare_data(processed_csv):
    df = pd.read_csv(processed_csv, parse_dates=['timestamp'])
    from features.features import make_features
    df = make_features(df)
    # target: next bar direction
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna().reset_index(drop=True)

    # Include risk-based features in X
    X = df[['rsi', 'atr', 'roc', 'ma_spread', 'pos_size_units', 'pos_used_leverage', 'pos_notional']]
    y = df['target']
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='processed CSV file')
    parser.add_argument('--model-out', type=str, default='models/xgb_model.joblib')
    args = parser.parse_args()
    X, y = prepare_data(args.input)
    model, score = train_xgb(X, y, model_path=args.model_out)
    print('Saved model to', args.model_out, 'best_score=', score)


if __name__ == '__main__':
    main()
