import joblib
import os
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score


def train_xgb(X, y, model_path='models/xgb_model.joblib'):
    tscv = TimeSeriesSplit(n_splits=5)
    best_model = None
    best_score = -1
    for tr_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        m = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        m.fit(X_tr, y_tr)
        preds = m.predict(X_val)
        score = accuracy_score(y_val, preds)
        if score > best_score:
            best_score = score
            best_model = m
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    return best_model, best_score
