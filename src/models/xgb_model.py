import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_xgb(X, y, model_path='models/xgb_model.joblib'):
    """
    Train an XGBoost classifier on given data and save the model.

    Args:
        X (pd.DataFrame): Features including risk-based and moving average spread.
        y (pd.Series): Target variable (e.g., next bar up/down).
        model_path (str): Path to save the trained model.

    Returns:
        model: Trained XGBClassifier model.
        score: Accuracy score on validation set.
    """
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize XGBoost classifier
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # Fit model
    model.fit(X_train, y_train)

    # Validate
    preds = model.predict(X_val)
    score = accuracy_score(y_val, preds)

    # Save model
    joblib.dump(model, model_path)

    return model, score

def load_model(model_path='models/xgb_model.joblib'):
    """
    Load a saved XGBoost model.

    Args:
        model_path (str): Path to the saved model file.

    Returns:
        Loaded model.
    """
    return joblib.load(model_path)

