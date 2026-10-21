from agent.state import PipelineState
from tools.feature_engineering import engineer_features

def feature_node(state: PipelineState) -> dict:
    """
    Applies imputation, encoding and scaling (fitted on the training split only).
    """
    X_train, X_test = engineer_features(state["X_train"], state["X_test"])
    
    # Overwrite the splits with the engineered versions for the training node
    return {"X_train": X_train, "X_test": X_test}
