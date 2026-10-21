from agent.state import PipelineState
from tools.mlflow_logger import log_experiment
import uuid

def evaluation_node(state: PipelineState) -> dict:
    """
    Evaluates metrics to pick the best model and logs the final result.
    """
    metrics = state["model_metrics"]
    task_type = state["data_profile"]["inferred_task_type"]
    
    best_model = None
    best_score = -float('inf') if task_type == 'classification' else float('inf')
    
    for model_name, scores in metrics.items():
        if task_type == 'classification':
            score = scores.get("f1_score")
            if score is None:
                continue  # no score for this model
            if score > best_score:
                best_score = score
                best_model = model_name
        else:
            score = scores.get("rmse")
            if score is None:
                continue  # no score for this model
            if score < best_score:
                best_score = score
                best_model = model_name
    
    if best_model is None:
        raise RuntimeError("No model produced a valid score; cannot select a best model.")
                
    best_model_info = {
        "name": best_model,
        "score": best_score,
        "metric": "f1_score" if task_type == "classification" else "rmse"
    }
    
    import shap
    import matplotlib.pyplot as plt
    import os
    
    # Log to MLflow
    run_name = f"AutoML_Run_{uuid.uuid4().hex[:8]}"
    log_experiment(run_name, task_type, metrics, best_model)
    
    # Generate SHAP Plot
    shap_plot_path = ""
    try:
        model_obj = state["trained_models"][best_model]
        X_test = state["X_test"]
        
        # Calculate SHAP values
        if best_model in ["Random Forest", "XGBoost"]:
            explainer = shap.TreeExplainer(model_obj, feature_perturbation="tree_path_dependent")
            shap_values = explainer(X_test)
        else:
            explainer = shap.Explainer(model_obj, X_test)
            shap_values = explainer(X_test)
        
        # Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        
        # Save
        dataset_name = state.get("dataset_name") or os.path.splitext(os.path.basename(state["raw_data_path"]))[0]
        shap_dir = os.path.join(state.get("output_dir", "outputs"), "shap_plots")
        os.makedirs(shap_dir, exist_ok=True)
        shap_plot_path = os.path.join(shap_dir, f"{dataset_name}_shap.png")
        plt.tight_layout()
        plt.savefig(shap_plot_path, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"SHAP generation failed: {e}")
        shap_plot_path = ""
        
    return {
        "best_model_info": best_model_info,
        "shap_plot_path": shap_plot_path
    }
