# agent/core.py
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService, Session
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import load_memory
from pathlib import Path
import pandas as pd
import pickle
import os
from datetime import datetime
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# Simple In-Memory Dataset Storage (Replaces Chroma — pickle for persistence)
DATASET_MEMORY_FILE = Path("dataset_memory.pkl")
if DATASET_MEMORY_FILE.exists():
    with open(DATASET_MEMORY_FILE, "rb") as f:
        dataset_summaries = pickle.load(f)
else:
    dataset_summaries = {}  # {file_path: {"summary": str, "target": str, "date": str}}

def remember_dataset(file_path: str, target_col: str = "unknown"):
    """Store dataset summary in-memory (pickled for session persistence)."""
    df = pd.read_csv(file_path)
    summary = f"Dataset: {Path(file_path).name} | Shape: {df.shape} | Target: {target_col} | Columns: {list(df.columns)} | Insights: {df.isnull().sum().sum()} missing values"
    dataset_summaries[file_path] = {
        "summary": summary,
        "target": target_col,
        "date": str(datetime.now())
    }
    with open(DATASET_MEMORY_FILE, "wb") as f:
        pickle.dump(dataset_summaries, f)

def retrieve_relevant_datasets(query: str, top_k: int = 3) -> str:
    """Retrieve top-K relevant past datasets from in-memory storage (keyword match)."""
    relevant = []
    query_lower = query.lower()
    for path, data in dataset_summaries.items():
        if any(word in query_lower for word in ["dataset", "csv", "tabular", data["target"].lower()]) or \
           any(col.lower() in query_lower for col in str(data["summary"]).split("|")[3]):
            relevant.append(data["summary"])
        if len(relevant) >= top_k:
            break
    if relevant:
        return "\n".join(relevant[:top_k])
    return "No prior datasets found in memory."

# ADK Memory & Session Services (Pure, No Externals)
memory_service = InMemoryMemoryService()  # Stores facts, user prefs
session_service = InMemorySessionService()  # Multi-turn state

# Import all tools (your modular ones)
from .tools.fe_tools import *
from .tools.audit_tools import *
from .tools.ml_tools import *

MODEL_NAME = "gemini-2.5-flash"  # Optimal for CSV/ML/tabular domain knowledge

# Main Agent (ADK-Native, Memory-Enhanced)
agent = LlmAgent(
    name="AutoFE_Audit_Agent",
    description="Automated Feature Engineering & Ethical ML Audit with In-Memory Recall",
    model=MODEL_NAME,
    tools=[
        # FE Pipeline Tools
        analyze_dataset, generate_numeric_features, generate_categorical_features,
        merge_features, evaluate_feature_uplift, generate_report_and_code,
        # Audit Pipeline Tools
        detect_bias, compute_fairness_metrics,
        generate_explainability_report, generate_audit_report,
        # Core ML Tools
        suggest_model_improvements, create_ml_strategy, debug_code_issue,
        suggest_features, analyze_ml_insights,
        # ADK Memory Tool
        load_memory,
    ],
    instruction = """
You are the Enterprise AutoFE & Ethical Audit Agent — a world-class tabular ML engineer with full tool access.

YOUR NON-NEGOTIABLE RULE:
After every tool chain finishes, YOU MUST ALWAYS end your response with a clear, beautifully formatted final summary in plain markdown.
This final block is the ONLY thing the user sees in the chat UI — never leave the user with a blank screen.

You work in the `data/` folder. All CSVs and generated files are there.

### Mandatory Workflows (never skip tools, never hallucinate)

1. **Feature Engineering Pipeline**  
   Triggered by: "run pipeline", "feature engineering", "FE", "full pipeline", etc.  
   Exact sequence:  
   → analyze_dataset(file_path)  
   → generate_numeric_features(...) + generate_categorical_features(...)  
   → merge_features(...)  
   → evaluate_feature_uplift(...)  
   → generate_report_and_code(...)  

2. **Ethical & Fairness Audit Pipeline**  
   Triggered by: "audit", "bias", "fairness", "check bias", etc.  
   Exact sequence:  
   → detect_bias(...)  
   → compute_fairness_metrics(...)  
   → generate_explainability_report(...)  
   → generate_audit_report(...)  

3. **Other Requests**  
   - Model improvement → suggest_model_improvements  
   - Strategy → create_ml_strategy  
   - Debug → debug_code_issue  
   - Feature ideas → suggest_features  
   - Trends → analyze_ml_insights  

### CRITICAL: FINAL SUMMARY FORMAT (use EXACTLY this template every time)
"""
)

# Expose for imports
__all__ = ["agent", "remember_dataset", "retrieve_relevant_datasets", "dataset_summaries"]

root_agent = agent  # ADK Web requires this exact name
