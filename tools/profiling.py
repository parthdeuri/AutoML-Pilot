import pandas as pd
from typing import Any

def profile_dataset(file_path: str, target_column: str) -> dict[str, Any]:

    df = pd.read_csv(file_path, sep=None, engine='python')
    
    stats = {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "columns": {}
    }
    
    for col in df.columns:
        col_series = df[col]
        dtype = str(col_series.dtype)
        num_missing = int(col_series.isnull().sum())
        missing_pct = (num_missing / len(df)) * 100
        num_unique = int(col_series.nunique())
        
        col_stat = {
            "dtype": dtype,
            "missing_count": num_missing,
            "missing_pct": missing_pct,
            "unique_count": num_unique
        }
        
        if pd.api.types.is_numeric_dtype(col_series):
            col_stat["mean"] = float(col_series.mean()) if num_missing < len(df) else None
            col_stat["min"] = float(col_series.min()) if num_missing < len(df) else None
            col_stat["max"] = float(col_series.max()) if num_missing < len(df) else None
        
        stats["columns"][col] = col_stat

    # Task Inference
    task_type = "regression"
    if target_column in df.columns:
        target_series = df[target_column]
        target_unique = target_series.nunique()
        target_dtype = target_series.dtype
        
        # If non-numeric (string/object/category), boolean, or few unique values, it's likely classification
        if (not pd.api.types.is_numeric_dtype(target_dtype)
                or pd.api.types.is_bool_dtype(target_dtype)
                or target_unique <= 20):
            task_type = "classification"
    
    stats["inferred_task_type"] = task_type
    stats["target_column"] = target_column

    return stats
