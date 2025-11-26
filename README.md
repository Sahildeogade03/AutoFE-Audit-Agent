# AutoFE-Audit-Agent
Agent that builds better features, checks for bias, and writes the pipeline code - all in plain English.
---

## Problem Statement

ML teams lose significant time to two repetitive tasks:

### 1. Manual Feature Engineering  
Transformations (logs, bins, encodings, interactions) require trial-and-error, lack automation, and rarely produce reusable pipelines.

### 2. Incomplete Ethical Audits  
Bias detection is manual or late in the process. Fairness metrics, bias checks, and explainability tools are scattered across multiple libraries.

### 🎯 Goal  
Build an ML Agent that:

- analyzes datasets  
- engineers & evaluates features  
- detects bias  
- computes fairness metrics  
- generates SHAP-style explanations  
- outputs production-ready Python code  

All triggered via *natural language instructions*.

---

## Why Agents?

Agents enable:

- **Sequential reasoning:** analyze → engineer → merge → evaluate → report  
- **Parallel reasoning:** fairness, explainability, research tasks running together  
- **Looping:** up to 15 tool iterations  
- **Short-term memory:** last 20 messages retained  
- **Quality control:** 6-point response validation  
- **Observability:** logging, metrics, error tracking  

Result → adaptive, auditable ML workflows instead of rigid scripts.

---

## Architecture Overview

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F16099101%2Fe76933d054c027f59eb6769840893ec5%2F1.png?generation=1764069301709225&alt=media)

---

# Tools & Components

## Core Reasoning Tools (5)

| Tool | Purpose | Description |
|------|---------|-------------|
| Model improvement suggestions | Optimization | Suggests better models, hyperparams, ensembles. |
| ML strategy creation | Planning | Structured ML action plans for custom goals. |
| Code debugging | Error fixing | Finds root causes & returns corrected code. |
| Feature suggestions | Ideation | Recommends new features, interactions, transformations. |
| Research/explanation generation | Knowledge | Summaries, conceptual insights, ML explanations. |

## Feature Engineering Pipeline Tools (6)

| Tool | Purpose | Description |
|------|---------|-------------|
| analyze_dataset | Profiling | Types, missing %, skew, cardinality, distributions. |
| generate_numeric_features | Numeric FE | Log, square, binning, numeric transforms. |
| generate_categorical_features | Categorical FE | One-hot, target, frequency encodings. |
| merge_features | Assembly | Combines numeric & categorical outputs. |
| evaluate_feature_uplift | Evaluation | Baseline vs. enhanced model uplifts. |
| generate_report_and_code | Deliverables | FE report + `feature_pipeline.py`. |

## Ethical Audit Tools (4)

| Tool | Purpose | Description |
|------|---------|-------------|
| detect_bias | Bias Detection | Flags demographic parity issues. |
| compute_fairness_metrics | Fairness | Equalized odds, parity diff, impact ratio. |
| generate_explainability_report | Explainability | SHAP-style summaries. |
| generate_audit_report | Final Audit | Combined bias + fairness + SHAP with remediation. |

---

# Workflows, Memory, Technicals & Impact

## 1. Workflows Enabled

### **A. End-to-End Feature Engineering**
Query: *"Auto-engineer features for Titanic."*  
Process: analyze → generate → merge → uplift → report → code  
Outputs:  
- `final_features.csv`  
- FE summary report  
- `feature_pipeline.py`

### **B. Ethical Audit Workflow**
Query: *"Audit for bias on Sex."*  
Tools: bias → fairness → explainability → audit report

### **C. Parallel Research Workflow**
Query: *"Explain bias, fairness, and explainability."*  
Multiple research tools combine into a unified output.

---

## 2. Memory, Logging & Quality

- **ConversationMemory (20 turns)** for context retention  
- **AgentLogger** for events, errors, runtimes  
- **Stats Dashboard** for performance tracking  
- **6-Point Response Validator** ensures clarity, examples, code correctness, and actionability  

---

## 3. Technical Highlights

- Up to **15-step tool-calling loops**  
- Works on **real datasets** 
- **NaN-safe** numeric transformations  
- Generates production-ready **FE and fairness pipelines**  
- SHAP & fairness are simulated but easily swappable with:  
  - AIF360  
  - Fairlearn  
  - SHAP  

---

## 4. Testing & Demos

Included examples demonstrate:

- Feature Engineering  
- Fairness Auditing  
- Explainability  
- Research Insights  
- Batch Query Execution  
- Observability Dashboard  

---

## 5. Future Improvements

- **Persistent Vector Memory**  
  Add Chroma/Pinecone → agent remembers every past dataset, feature uplift, and bias result forever

- **Reviewer Agent (A2A)**  
  The second agent auto-critiques every run: “Drop Age_sq—it hurts fairness by 12%”

- **One-Click MLflow + DVC Export**  
  `feature_pipeline.py` + data versions auto-registered as MLflow stage & DVC-tracked

- **Real AIF360/Fairlearn Integration**  
  Swap simulated metrics for production-grade bias mitigation & re-evaluation loops

- **Auto Dataset & Model Cards**  
  Generate HF-style dataset/model cards and bias badge after every run (compliance-ready) 

---

## 6. Impact

This agent compresses hours of manual ML work into a single conversational workflow by combining:

- sequential, parallel & looped agent reasoning  
- custom ML tools  
- integrated fairness + explainability  
- automatic, reproducible code generation  

An **enterprise-ready ML copilot** for any structured dataset.

<img width="1919" height="737" alt="Screenshot 2025-11-25 121107" src="https://github.com/user-attachments/assets/103f6930-2061-49ac-a0ff-c0818eac79a9" />

<img width="1918" height="872" alt="Screenshot 2025-11-25 115805" src="https://github.com/user-attachments/assets/fa449d81-cce3-4cf0-80bf-35a84bbd21a8" />

<img width="1918" height="871" alt="Screenshot 2025-11-25 115833" src="https://github.com/user-attachments/assets/9c3f7ca3-22c4-4fb1-af6e-8f2a58cc7d13" />

<img width="1919" height="769" alt="Screenshot 2025-11-25 120323" src="https://github.com/user-attachments/assets/64469c9d-5e92-4756-ae05-09106e03f9e8" />

<img width="1919" height="960" alt="Screenshot 2025-11-25 120002" src="https://github.com/user-attachments/assets/a19e0f62-8b2b-4582-9f29-4a664199af79" />
