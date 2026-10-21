import os
import sys
import warnings
import logging
from dotenv import load_dotenv

# Suppress all warnings and unnecessary logs
warnings.filterwarnings("ignore")
logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)

# Load environment variables (API keys)
load_dotenv()

# Add the project root to sys.path to resolve 'agent' module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
import pandas as pd
from agent.graph import build_graph

def run_benchmark():
    """
    Runs the AutoML-Pilot against datasets.
    Can specify a single CSV and target, or run all datasets in 'datasets/' folder.
    """
    parser = argparse.ArgumentParser(description="Run AutoML-Pilot Benchmark")
    parser.add_argument("--csv", type=str, help="Specific CSV filename in the datasets folder (e.g. titanic.csv)", default=None)
    parser.add_argument("--target", type=str, help="Target column name (optional)", default="")
    args = parser.parse_args()

    datasets_dir = "datasets"
    if not os.path.exists(datasets_dir):
        print(f"Directory {datasets_dir} does not exist. Creating it.")
        os.makedirs(datasets_dir)
        print("Please add CSV datasets to the directory and run again.")
        return
        
    if args.csv:
        csv_files = [args.csv]
    else:
        csv_files = [f for f in os.listdir(datasets_dir) if f.endswith(".csv")]
        
    if not csv_files:
        print("No CSV files found to process.")
        return
        
    app = build_graph()
    
    for csv_file in csv_files:
        print(f"\n{'='*50}\nStarting benchmark for: {csv_file}\n{'='*50}")
        file_path = os.path.join(datasets_dir, csv_file)
        
        try:
            initial_state = {
                "raw_data_path": file_path,
                "target_column": args.target,
                "dataset_name": Path(csv_file).stem
            }
            final_state = app.invoke(initial_state)
            
            print(f"SUCCESS: {csv_file}")
            print(f"Best Model: {final_state['best_model_info']['name']} (Score: {final_state['best_model_info']['score']:.4f})")
            
            # Save the report
            report_path = os.path.join("outputs", "model_cards", f"{Path(csv_file).stem}_report.md")
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(final_state["report"])
            print(f"Report saved to: {report_path}")
            
        except Exception as e:
            print(f"FAILED: {csv_file}. Error: {e}")

if __name__ == "__main__":
    run_benchmark()
