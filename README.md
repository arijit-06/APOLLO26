# ChronoDrift-AI 🚀
**SIH 2026 Problem Statement ID: 26170 | Team: Apollo 26**

---

## 🛰 Overview & Aerospace Impact
During 125°C accelerated burn-in testing for space-grade semiconductor payloads, traditional static failure thresholds often miss microscopic drift. These **latent defects** technically pass the screening phase but carry degradation curves that inevitably lead to catastrophic burnout in the vacuum of space.

**ChronoDrift-AI** is a comprehensive, full-stack temporal AI engine built to solve this critical vulnerability. By ingesting discrete ATE logs (0h, 24h, 96h, 168h) and mapping them into continuous trajectories, our 3-Tier Machine Learning Core identifies subtle sequence acceleration, instantly isolating components destined to fail in orbit.

## 🛠 Comprehensive Tech Stack
*   **Machine Learning / AI Core:** Python, PyTorch, Scikit-Learn, SciPy, SHAP, Captum (Integrated Gradients)
*   **Backend & Data Pipeline:** FastAPI, Uvicorn, Pandas, NumPy, Pydantic
*   **Frontend Command Center:** React.js, Vite, Tailwind CSS, Apache ECharts, Lucide Icons

## 📂 Project Structure
```text
ChronoDrift-AI/
├── backend/
│   ├── main.py
│   ├── routes.py
│   └── ml_runner.py
├── frontend/
│   ├── package.json
│   ├── src/
│   │   ├── App.jsx
│   │   └── components/
│   │       ├── Dashboard.jsx
│   │       ├── UploadATE.jsx
│   │       ├── WaferMap.jsx
│   │       ├── DriftCharts.jsx
│   │       └── ExplainabilityPanel.jsx
├── ml_core/
│   ├── ml_preprocessing.py
│   ├── tier1_statistical.py
│   ├── tier2_autoencoder.py
│   └── tier3_explainability.py
├── data/
│   ├── data_synthesis.py
│   └── burn_in_sequences.npy
├── test_e2e_simulation.py
└── README.md
```

## 📊 Implementation of Proposed Solutions (PPT Alignment)
Our codebase natively reflects every conceptual claim made in our SIH Idea Presentation:

| PPT Presentation Claim | Codebase Implementation Mapping |
| :--- | :--- |
| **"0h, 24h, 96h, 168h Time Alignment"** | `ml_core/ml_preprocessing.py` (PCHIP Interpolation & `parse_csv_to_tensor`) |
| **"Dynamic Multi-Parameter Correlation & Lot Normalization"** | Intra-lot robust DPAT Z-score scaling applied iteratively per batch |
| **"Tier 1 Statistical Filter"** | `ml_core/tier1_statistical.py` (PCA & Mahalanobis Distance for gross fails) |
| **"Tier 2 Deep Reconstruction"** | `ml_core/tier2_autoencoder.py` (LSTM-Autoencoder for drift trajectory loss) |
| **"Explainable AI (XAI)"** | `ml_core/tier3_explainability.py` (Captum Integrated Gradients for root-cause attribution) |

## 🚀 Local Setup Instructions

### 1. Initialize the FastAPI Backend
```bash
# Clone the repository
git clone https://github.com/your-username/ChronoDrift-AI.git
cd ChronoDrift-AI

# Install Python requirements
pip install fastapi uvicorn torch scikit-learn pandas scipy captum aiofiles

# Boot the asynchronous integration bridge
python backend/main.py
```
*The ML API will successfully bind to `http://localhost:8000`.*

### 2. Launch the React Dashboard
```bash
# Open a new terminal and navigate to the UI directory
cd frontend

# Install Node dependencies
npm install recharts lucide-react tailwindcss

# Boot the Vite development server
npm run dev
```
*The interactive ISRO command center will launch on `http://localhost:3000`.*

## 👥 Team Apollo 26 Credits
*   **Arijit** – Lead AI / ML Architecture (Track A)
*   **Sourashis Sabud** – Full-Stack / API Integration Engineer (Track B)
*   **Protyay Saha** – Frontend Architecture & UI/UX (Track B)

*Engineered with precision for the Smart India Hackathon 2026.*
