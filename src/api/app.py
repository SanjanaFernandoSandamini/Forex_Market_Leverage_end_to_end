from flask import Flask, jsonify, request
import joblib
import os
import pandas as pd

app = Flask(__name__)
MODEL_PATH = os.getenv('MODEL_PATH', 'models/xgb_model.joblib')

model = None

print(f"Looking for model file at: {MODEL_PATH}")
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print("Model file not found.")
@app.route('/')
def home():
    return jsonify({
        'message': 'Forex Leverage API is running',
        'endpoints': ['/health', '/predict']
    })

@app.route('/health')
def health():
    return jsonify({'status':'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error':'model not found'}), 500

    payload = request.json
    if not payload:
        return jsonify({'error':'empty payload'}), 400

    try:
        df = pd.DataFrame(payload)
        preds = model.predict(df)
        return jsonify({'preds': preds.tolist()})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
