import streamlit as st
import os
import re
import uuid
import warnings
import logging
from pathlib import Path
from dotenv import load_dotenv
from tools.sql_chat import ask_dataset
from agent.graph import build_graph

# Suppress all warnings and unnecessary logs
warnings.filterwarnings("ignore")
logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)

# Load environment variables (API keys)
load_dotenv()

st.set_page_config(page_title="AutoML-Pilot", page_icon="🚀", layout="wide")

st.title("AutoML-Pilot")
st.subheader("Navigating your machine learning pipelines autonomously.")

st.markdown("""
Upload a CSV dataset, specify the target column, and let the agent autonomously clean the data, engineer features, train models, and generate a final model card!
""")

# Each browser session gets a private workspace so visitors never see each other's files
PROJECT_ROOT = Path(__file__).resolve().parent

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

WORKSPACE = PROJECT_ROOT / "workspace" / st.session_state.session_id
UPLOADS_DIR = WORKSPACE / "uploads"
MODEL_CARDS_DIR = WORKSPACE / "model_cards"
SHAP_DIR = WORKSPACE / "shap_plots"
for d in (UPLOADS_DIR, MODEL_CARDS_DIR, SHAP_DIR):
    d.mkdir(parents=True, exist_ok=True)


def safe_stem(name: str) -> str:
    """File name without extension, with unsafe characters replaced by '_'."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", Path(name).stem) or "dataset"


# Initialize session state for Chat
if "active_chat_dataset" not in st.session_state:
    st.session_state.active_chat_dataset = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "dataset_path_for_chat" not in st.session_state:
    st.session_state.dataset_path_for_chat = None

if st.session_state.active_chat_dataset:
    st.button("🔙 Back to Report", on_click=lambda: st.session_state.update({"active_chat_dataset": None}))
    st.header(f"💬 Chat with Dataset: {st.session_state.active_chat_dataset}")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # Input box
    if prompt := st.chat_input("Ask a question about the data (e.g., 'What is the average age?'):"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Executing SQL query..."):
                response = ask_dataset(st.session_state.dataset_path_for_chat, prompt)
                st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

else:
    tab1, tab2 = st.tabs(["Run Pipeline", "History"])

    with tab1:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        target_column = st.text_input("Target Column Name (Optional)", value="", help="Leave blank if you want the agent to guess.")
    
        if uploaded_file is not None:
            # Save the upload in this session's workspace so chat and history can see it
            dataset_name = safe_stem(uploaded_file.name)
            dataset_path = UPLOADS_DIR / f"{dataset_name}.csv"
            dataset_path.write_bytes(uploaded_file.getvalue())
                
            col1, col2 = st.columns(2)
            with col1:
                run_pipeline = st.button("Run Pipeline")
            with col2:
                if st.button("💬 Chat with this Dataset", key="chat_tab1"):
                    st.session_state.active_chat_dataset = dataset_name
                    st.session_state.dataset_path_for_chat = str(dataset_path)
                    st.session_state.chat_history = []
                    st.rerun()
            
            st.markdown("---")
            if run_pipeline:
                with st.spinner("Agent is working..."):
                    try:
                        # Initialize Graph
                        app = build_graph()
                        
                        # Initial State (runs directly on the saved upload)
                        initial_state = {
                            "raw_data_path": str(dataset_path),
                            "target_column": target_column.strip(),
                            "dataset_name": dataset_name,
                            "output_dir": str(WORKSPACE)
                        }
                        
                        # Run the graph
                        final_state = app.invoke(initial_state)
                        
                        st.success("Pipeline execution complete!")
                        
                        # Save the report
                        report_content = final_state.get("report", "No report generated.")
                        report_path = MODEL_CARDS_DIR / f"{dataset_name}_report.md"
                        with open(report_path, "w", encoding="utf-8") as f:
                            f.write(report_content)
                        
                        # Display Results
                        st.markdown(report_content)
                        
                        shap_path = final_state.get("shap_plot_path")
                        if shap_path and os.path.exists(shap_path):
                            st.subheader("Feature Importance (SHAP)")
                            st.image(shap_path)
                        
                        with st.expander("View Raw Cleaning Plan"):
                            st.json(final_state.get("cleaning_plan", {}))
                            
                        with st.expander("View Model Metrics"):
                            st.json(final_state.get("model_metrics", {}))
                            
                    except Exception as e:
                        st.error(f"An error occurred during execution: {e}")
    
    with tab2:
        st.header("Past Model Cards")
        reports_dir = MODEL_CARDS_DIR
        if reports_dir.exists():
            reports = sorted([f for f in os.listdir(reports_dir) if f.endswith("_report.md")])
            if reports:
                selected_report = st.selectbox("Select a model card", reports)
                if selected_report:
                    report_path = reports_dir / selected_report
                    with open(report_path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    
                    # Infer dataset name
                    history_name = selected_report.replace("_report.md", "")
                    dataset_path = UPLOADS_DIR / f"{history_name}.csv"
                    
                    if dataset_path.exists():
                        st.info(f"Dataset Path: `{dataset_path}`")
                        if st.checkbox("Show Dataset Sample", key=selected_report):
                            import pandas as pd
                            df = pd.read_csv(dataset_path, sep=None, engine='python', nrows=100)
                            st.dataframe(df)
                    else:
                        st.warning(f"Associated dataset ({history_name}.csv) not found in this session's uploads.")
                        
                    # Display Report
                    st.markdown(content)
                    
                    # Try to load corresponding SHAP plot
                    shap_path = SHAP_DIR / f"{history_name}_shap.png"
                    if shap_path.exists():
                        st.subheader("Feature Importance (SHAP)")
                        st.image(str(shap_path))
                        
                    st.markdown("---")
                    if st.button("💬 Chat with this Dataset", key=f"chat_tab2_{selected_report}"):
                        st.session_state.active_chat_dataset = history_name
                        st.session_state.dataset_path_for_chat = str(dataset_path)
                        st.session_state.chat_history = []
                        st.rerun()
            else:
                st.info("No past model cards found.")
        else:
            st.info("No past model cards found.")
