"""
Training entrypoint. Train an XGBoost classifier predicting next-bar sign of return,
including risk-based features in training data.
"""

import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from models.xgb_model import train_xgb
from features.features import make_features


def prepare_data(processed_csv, scale_features=True):
    df = pd.read_csv(processed_csv, parse_dates=['timestamp'])
    
    # Generate features (including risk and moving average features)
    df = make_features(df)
    
    # Target: next bar price increase (binary)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    df = df.dropna().reset_index(drop=True)
    
    feature_cols = ['rsi', 'atr', 'roc', 'ma_spread', 'pos_size_units', 'pos_used_leverage', 'pos_notional']
    X = df[feature_cols]
    y = df['target']

    if scale_features:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)
    
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to processed CSV file')
    parser.add_argument('--model-out', type=str, default='models/xgb_model.joblib', help='Output model file path')
    args = parser.parse_args()
    
    X, y = prepare_data(args.input)
    model, score = train_xgb(X, y, model_path=args.model_out)
    
    print(f'Saved model to {args.model_out} with best score: {score:.4f}')


if __name__ == '__main__':
    main()
