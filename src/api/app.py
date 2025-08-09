from flask import Flask, jsonify, request
import joblib
import os
import pandas as pd
from src.features.features import make_features  # your feature engineering function

app = Flask(__name__)
MODEL_PATH = os.getenv('MODEL_PATH', 'models/xgb_model.joblib')

model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'model not found'}), 500

    try:
        payload = request.get_json(force=True)
        if not isinstance(payload, list):
            return jsonify({'error': 'JSON payload must be a list of records'}), 400

        df = pd.DataFrame(payload)

        # Basic validation: check required columns are present for features
        required_cols = ['timestamp', 'open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return jsonify({'error': f'Missing columns in input data: {missing_cols}'}), 400

        # Ensure timestamp is datetime type
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Apply your feature engineering pipeline (which adds technical + risk features)
        df_features = make_features(df)

        # Select columns your model expects (same as training)
        feature_cols = ['rsi', 'atr', 'roc', 'ma_spread', 'pos_size_units', 'pos_used_leverage', 'pos_notional']
        if not all(col in df_features.columns for col in feature_cols):
            return jsonify({'error': 'Required feature columns missing after processing'}), 500

        X = df_features[feature_cols]

        preds = model.predict(X)
        return jsonify({'preds': preds.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Run with debug=False for production-like behavior
    app.run(host='0.0.0.0', port=5000, debug=False)
