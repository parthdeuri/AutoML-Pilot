import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer

MAX_ONEHOT_CATEGORIES = 15


def _to_str_object(X):
    # Plain-object strings with NaN kept for missing values, so imputers/encoders behave the same on any pandas dtype
    X = pd.DataFrame(X)
    out = X.astype(str).astype(object)
    out[X.isna()] = np.nan
    return out


def engineer_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Imputes, scales and encodes features. All transformers are fitted on X_train only
    and then applied to X_test (no leakage).
    """
    if X_train.shape[1] == 0:
        raise ValueError("No feature columns left after cleaning (all columns were dropped).")

    bool_cols, num_cols, onehot_cols, ordinal_cols = [], [], [], []
    for col in X_train.columns:
        if pd.api.types.is_bool_dtype(X_train[col]):
            bool_cols.append(col)
        elif pd.api.types.is_numeric_dtype(X_train[col]):
            num_cols.append(col)
        elif X_train[col].nunique() <= MAX_ONEHOT_CATEGORIES:
            onehot_cols.append(col)
        else:
            ordinal_cols.append(col)

    # Safe categorical imputer: strings + "Missing" for NaN
    def cat_steps():
        return [
            ("to_str", FunctionTransformer(_to_str_object, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ]

    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num_cols))
    if bool_cols:
        transformers.append(("bool", FunctionTransformer(lambda X: pd.DataFrame(X).astype(float), feature_names_out="one-to-one"), bool_cols))
    if onehot_cols:
        transformers.append(("onehot", Pipeline(cat_steps() + [
            ("encode", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ]), onehot_cols))
    if ordinal_cols:
        transformers.append(("ordinal", Pipeline(cat_steps() + [
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), ordinal_cols))

    ct = ColumnTransformer(transformers, verbose_feature_names_out=False)
    ct.set_output(transform="pandas")

    train_out = ct.fit_transform(X_train).astype(float)
    test_out = ct.transform(X_test).astype(float)
    return train_out, test_out
