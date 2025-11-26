# agent/tools/fe_tools.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pathlib import Path
import json
from ..agent import remember_dataset

def analyze_dataset(file_path: str) -> str:
    df = pd.read_csv(file_path)
    analysis = {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "skew": df.skew(numeric_only=True).round(2).to_dict(),
        "cardinality": df.select_dtypes(include="object").nunique().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict()
    }
    remember_dataset(file_path, "unknown")
    return json.dumps(analysis, indent=2)

def generate_numeric_features(file_path: str, columns: str) -> str:
    df = pd.read_csv(file_path)
    cols = [c.strip() for c in columns.split(",")]
    for col in cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[f"{col}_log"] = np.log1p(df[col].abs())
            df[f"{col}_sq"] = df[col] ** 2
            df[f"{col}_bin"] = pd.qcut(df[col], q=5, duplicates="drop", labels=False)
    Path("data").mkdir(exist_ok=True)
    out = "data/num_features.csv"
    df.to_csv(out, index=False)
    return f"Numeric features saved → {out}"

def generate_categorical_features(file_path: str, columns: str) -> str:
    df = pd.read_csv(file_path)
    cols = [c.strip() for c in columns.split(",")]
    encoded = pd.get_dummies(df[cols], drop_first=True, dtype=int)
    out = "data/cat_features.csv"
    encoded.to_csv(out, index=False)
    return f"Categorical features saved → {out}"

def merge_features(numeric_file: str, categorical_file: str) -> str:
    df1 = pd.read_csv(numeric_file)
    df2 = pd.read_csv(categorical_file)
    merged = pd.concat([df1, df2], axis=1)
    out = "data/final_features.csv"
    merged.to_csv(out, index=False)
    return f"Merged dataset → {out} ({merged.shape})"

def evaluate_feature_uplift(base_file: str, new_file: str, target_col: str) -> str:
    base = pd.read_csv(base_file)
    new = pd.read_csv(new_file)
    Xb, yb = base.drop(columns=[target_col], errors="ignore"), base[target_col]
    Xn, yn = new.drop(columns=[target_col], errors="ignore"), new[target_col]

    model = LogisticRegression(max_iter=1000)
    base_acc = accuracy_score(yb, model.fit(Xb, yb).predict(Xb))
    new_acc = accuracy_score(yn, model.fit(Xn, yn).predict(Xn))
    uplift = new_acc - base_acc
    return f"Base: {base_acc:.4f} → New: {new_acc:.4f} | Uplift: {uplift:+.4f}"

def generate_report_and_code(analysis_json: str, feature_files: str) -> str:
    report = f"# Feature Engineering Report\nGenerated {len(feature_files.split(','))} files.\nPipeline ready."
    code = '# feature_pipeline.py\n# Auto-generated — run with: python feature_pipeline.py'
    Path("data/feature_report.md").write_text(report)
    Path("data/feature_pipeline.py").write_text(code)
    return "Report + pipeline code generated → data/feature_report.md & feature_pipeline.py"