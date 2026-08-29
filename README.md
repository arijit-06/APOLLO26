# PBAA - Pre-flight Burn-In Anomaly Analysis 🚀
**SIH 2026 Problem Statement ID: 26170 | Team: Apollo 26**

---

## 🛰 Overview & Aerospace Impact
During 125°C accelerated burn-in testing for space-grade semiconductor payloads, traditional static failure thresholds often miss microscopic drift. These **latent defects** technically pass the screening phase but carry degradation curves that inevitably lead to catastrophic burnout in the vacuum of space.

**PBAA - Pre-flight Burn-In Anomaly Analysis** is a comprehensive, full-stack temporal AI engine built to solve this critical vulnerability. By ingesting discrete ATE logs (0h, 24h, 96h, 168h) and mapping them into continuous trajectories, our 3-Tier Machine Learning Core identifies subtle sequence acceleration, instantly isolating components destined to fail in orbit.

## 🛠 Comprehensive Tech Stack
*   **Machine Learning / AI Core:** Python, PyTorch, Scikit-Learn, SciPy, SHAP, Captum (Integrated Gradients)
*   **Unified Dashboard:** Streamlit, Plotly, Pandas, NumPy

## 📂 Project Structure
```text
PBAA - Pre-flight Burn-In Anomaly Analysis/
├── app.py                     # Streamlit Unified Command Center
├── ml_runner.py               # Core Pipeline Orchestrator
├── ml_core/
│   ├── ml_preprocessing.py
│   ├── tier1_statistical.py
│   ├── tier2_autoencoder.py
│   └── tier3_explainability.py
└── data/
    ├── raw/                   # User Uploaded ATE Logs
    ├── processed/             # Intermediate Scaling Tensors
    ├── reports/               # SHAP JSON Explanations
    └── synthetic/             # Physics Simulation Engine
```

## 🏗 The 3-Tier Architecture
| PPT Presentation Claim | Codebase Implementation Mapping |
| :--- | :--- |
| **"0h, 24h, 96h, 168h Time Alignment"** | `ml_core/ml_preprocessing.py` (PCHIP Interpolation & `parse_csv_to_tensor`) |
| **"Dynamic Multi-Parameter Correlation & Lot Normalization"** | Intra-lot robust DPAT Z-score scaling applied iteratively per batch |
| **"Tier 1 Statistical Filter"** | `ml_core/tier1_statistical.py` (PCA & Mahalanobis Distance for gross fails) |
| **"Tier 2 Deep Reconstruction"** | `ml_core/tier2_autoencoder.py` (LSTM-Autoencoder for drift trajectory loss) |
| **"Explainable AI (XAI)"** | `ml_core/tier3_explainability.py` (Captum Integrated Gradients for root-cause attribution) |

## 🚀 Local Setup Instructions

### 1. Install Dependencies
```bash
# Clone the repository
git clone https://github.com/your-username/PBAA - Pre-flight Burn-In Anomaly Analysis.git
cd PBAA - Pre-flight Burn-In Anomaly Analysis

# Install Python requirements
pip install streamlit plotly pandas numpy torch scikit-learn scipy captum
```

### 2. Launch the Streamlit Dashboard
```bash
# Boot the Unified Command Center
streamlit run app.py
```
*The interactive ISRO command center will launch on `http://localhost:8501`.*

*Engineered with precision for the Smart India Hackathon 2026.*
