from agent.state import PipelineState
from tools.modeling import train_and_evaluate

def training_node(state: PipelineState) -> dict:
    """
    Trains the baseline models and collects their metrics (and any per-model errors).
    """
    task_type = state["data_profile"]["inferred_task_type"]
    
    metrics, models, errors = train_and_evaluate(
        state["X_train"], state["X_test"], state["y_train"], state["y_test"], task_type
    )
    
    return {
        "model_metrics": metrics,
        "trained_models": models,
        "model_errors": errors
    }
