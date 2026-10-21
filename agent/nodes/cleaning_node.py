import pandas as pd
from agent.state import PipelineState
from tools.cleaning import clean_data
from tools.modeling import split_dataset

def cleaning_node(state: PipelineState) -> dict:
    """
    Splits the raw data into train/test, then executes the JSON cleaning plan
    (fill values are learned from the training split only).
    """
    raw_path = state["raw_data_path"]
    plan = state["cleaning_plan"]
    target_col = state["data_profile"]["target_column"]
    task_type = state["data_profile"]["inferred_task_type"]
    
    df = pd.read_csv(raw_path, sep=None, engine='python')
    X_train, X_test, y_train, y_test = split_dataset(df, target_col, task_type)
    X_train, X_test = clean_data(X_train, X_test, plan)
    
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
