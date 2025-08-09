FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire src folder preserving package structure
COPY src ./src

# Expose the port your Flask app runs on
EXPOSE 5000

# Set PYTHONPATH so imports like `from src.models.xgb_model import ...` work
ENV PYTHONPATH=/app

# Run the Flask app
CMD ["python", "src/api/app.py"]


