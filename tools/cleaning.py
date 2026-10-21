import pandas as pd
from typing import Any

def clean_data(X_train: pd.DataFrame, X_test: pd.DataFrame, cleaning_plan: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies the cleaning plan. Fill values are computed from X_train only and applied to both frames.
    Actions on the target column are skipped because the target is no longer part of X.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()
    actions = cleaning_plan.get("actions", [])

    for act in actions:
        action_type = act.get("action")
        col = act.get("column")

        if col not in X_train.columns:
            continue

        if action_type == "drop_column":
            X_train = X_train.drop(columns=[col])
            X_test = X_test.drop(columns=[col])

        elif action_type == "impute":
            strategy = act.get("strategy")
            fill_value = None

            if strategy == "median" and pd.api.types.is_numeric_dtype(X_train[col]):
                fill_value = X_train[col].median()
            elif strategy == "mean" and pd.api.types.is_numeric_dtype(X_train[col]):
                fill_value = X_train[col].mean()
            elif strategy == "mode":
                mode = X_train[col].mode()
                fill_value = mode[0] if not mode.empty else None
            elif strategy == "constant":
                fill_value = act.get("value", "Missing")

            if fill_value is not None and not pd.isna(fill_value):
                X_train[col] = X_train[col].fillna(fill_value)
                X_test[col] = X_test[col].fillna(fill_value)

    return X_train, X_test
