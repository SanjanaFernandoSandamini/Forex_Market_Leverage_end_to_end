FROM python:3.11-slim

WORKDIR /app

# Copy only requirements first for caching
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy models directory
COPY models ./models

# Copy src directory
COPY src ./src

# Expose the port Flask uses
EXPOSE 5000

# Set environment variable for model path explicitly (optional but safer)
ENV MODEL_PATH=./models/xgb_model.joblib

# Run Flask app
CMD ["python", "src/api/app.py"]

