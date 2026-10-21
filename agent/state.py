from typing import TypedDict, Any, Optional

class PipelineState(TypedDict):
    """
    Represents the state of the machine learning pipeline as it moves through the LangGraph nodes.
    """
    # Raw Data Path provided by the user
    raw_data_path: str
    
    # Optional target column provided by the user (if unknown, empty string)
    target_column: str
    
    # Deterministic statistical summary extracted by the Profiler tool
    data_profile: dict[str, Any]
    
    # JSON-based plan for data cleaning, produced by the Planner LLM
    cleaning_plan: dict[str, Any]
    
    # Stable name of the dataset (file stem), used to name outputs
    dataset_name: str

    # Folder where outputs (e.g. SHAP plots) are written; defaults to "outputs"
    output_dir: str

    # Train/test split (features and target), produced by the Cleaner and then transformed by the Feature node
    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any
    
    # Information regarding scaling and categorical encoding
    feature_engineering_info: dict[str, Any]
    
    # Metrics logged during the training of various baseline models
    model_metrics: dict[str, Any]
    
    # Information on the top performing model
    best_model_info: dict[str, Any]
    
    # Dictionary of trained model objects
    trained_models: dict[str, Any]
    
    # Models that failed to train: {model name: error message}
    model_errors: dict[str, str]
    
    # Path to the generated SHAP feature importance plot (if generated)
    shap_plot_path: str
    
    # Final synthesized Markdown report
    report: str
