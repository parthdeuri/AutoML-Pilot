import math
import pandas as pd
from typing import Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score


def split_dataset(df: pd.DataFrame, target_column: str, task_type: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Drops rows with a missing target, then splits 80/20 BEFORE any imputation/scaling/encoding
    so nothing is learned from the test set.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
    if task_type not in ("classification", "regression"):
        raise ValueError("task_type must be either 'classification' or 'regression'")

    df = df.dropna(subset=[target_column]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No rows left: target column '{target_column}' is entirely missing.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    stratify = None
    if task_type == 'classification':
        # XGBoost requires 0-indexed integer labels
        y = pd.Series(LabelEncoder().fit_transform(y.astype(str)), name=target_column)
        # Stratify only if every class has at least 2 rows
        if y.value_counts().min() >= 2:
            stratify = y

    if len(df) < 2:
        raise ValueError("Need at least 2 rows with a valid target to split the data.")

    # Keep the test split non-empty for tiny datasets
    test_size = max(0.2, 1 / len(df)) if len(df) < 5 else 0.2
    if stratify is not None and math.ceil(len(df) * test_size) < y.nunique():
        stratify = None  # test split too small to hold every class

    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)


def train_and_evaluate(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
                       task_type: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """
    Trains baseline models. A failing model is recorded in `errors` instead of stopping the run.
    """
    if task_type == 'classification':
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
        }
    elif task_type == 'regression':
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(random_state=42),
            'XGBoost': XGBRegressor(random_state=42)
        }
    else:
        raise ValueError("task_type must be either 'classification' or 'regression'")

    metrics = {}
    trained = {}
    errors = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            if task_type == 'classification':
                metrics[name] = {
                    'accuracy': accuracy_score(y_test, preds),
                    'f1_score': f1_score(y_test, preds, average='weighted')
                }
            else:
                metrics[name] = {
                    'rmse': mean_squared_error(y_test, preds) ** 0.5,
                    'r2_score': r2_score(y_test, preds)
                }
            trained[name] = model
        except Exception as e:
            errors[name] = str(e)
            print(f"Model '{name}' failed: {e}")

    if not metrics:
        raise RuntimeError(f"All models failed to train: {errors}")

    return metrics, trained, errors
