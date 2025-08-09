from flask import Flask, jsonify, request
import joblib
import os
import pandas as pd

app = Flask(__name__)
MODEL_PATH = os.getenv('MODEL_PATH', 'models/xgb_model.joblib')

model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

@app.route('/health')
def health():
    return jsonify({'status':'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error':'model not found'}), 500
    payload = request.json
    df = pd.DataFrame(payload)
    preds = model.predict(df)
    return jsonify({'preds': preds.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
