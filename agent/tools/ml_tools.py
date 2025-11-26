# agent/tools/ml_tools.py

def suggest_model_improvements(model_type: str, current_score: float, target_score: float) -> str:
    return f"Switch to CatBoost + hyperopt. Expected lift: {target_score - current_score:+.3f}"

def create_ml_strategy(goal: str, timeframe_days: int, current_position: str) -> str:
    return f"Phase 1 ({timeframe_days//3} days): Data prep\nPhase 2: Modeling\nPhase 3: MLOps"

def debug_code_issue(error_message: str, code_context: str, framework: str = "sklearn") -> str:
    return "Fixed: Use .iloc instead of .loc with integer positions."

def suggest_features(dataset_description: str, target_variable: str, current_features: str) -> str:
    return "Add: family_size, title_extracted, fare_per_person, deck"

def analyze_ml_insights(topic: str) -> str:
    return f"Latest on {topic}: Gradient boosting still dominates tabular data (2025)."