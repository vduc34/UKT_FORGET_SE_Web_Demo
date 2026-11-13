# UKT × FORGET-SE: Adaptive Learning Path Recommendation System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20UI-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-FORGET--SE-green)]()

An end-to-end intelligent adaptive learning system that leverages **Uncertainty-aware Knowledge Tracing (UKT)**, models **memory decay** (Forgetting Curve) and **inter-concept interference**, and dynamically generates personalized remedial learning pathways across 10 Software Engineering concepts.

---

## 📌 Key Highlights

- **Uncertainty-Aware Knowledge Tracing (UKT):** Goes beyond standard KT models (DKT/AKT) by deriving calibrated covariance uncertainty proxies alongside mastery probability $P(\text{correct})$, preventing overconfidence on lucky guesses.
- **Forgetting & Interference Dynamics:** Combines logarithmic time-decay elapsed intervals with an empirical inter-concept interference matrix to penalize mastery scores under knowledge interference.
- **Adaptive Pedagogical Policy:** Implements a 3-tier action engine:
  - 🔵 **DIAGNOSE:** High prediction uncertainty $\rightarrow$ triggers targeted diagnostic questions.
  - 🟡 **REVIEW:** Memory decay or low safe mastery $\rightarrow$ schedules immediate spaced reinforcement.
  - 🟢 **ADVANCE:** High confidence and high safe mastery $\rightarrow$ clears prerequisite for forward progression.
- **Curriculum-Aware Path Planner:** Evaluates prerequisite chains according to course order and automatically constructs personalized remedial sequences.
- **Interactive Full-Stack Web Demo:** 6-module interactive Streamlit application with real-time sequence simulation, candidate evaluation, and curriculum exploration.

---

## 📊 Experimental Results & Metrics

Trained and evaluated on the **FORGET-SE** dataset (186 university students, 10,873 quiz interactions, 56 unique questions across 10 core Software Engineering Knowledge Components):

| Metric | Score | Description |
| :--- | :---: | :--- |
| **Validation AUC** | **0.7342** | Best epoch: 54 |
| **Validation Accuracy** | **66.68%** | Binary correctness prediction |
| **Validation Brier Score** | **0.2109** | Well-calibrated probabilistic predictions |
| **Validation Loss** | **0.6414** | Binary cross-entropy with uncertainty regularizer |

### The 10 Software Engineering Knowledge Components (KCs):
1. **KC1:** Git / Version Control
2. **KC2:** Design Patterns
3. **KC3:** Software Testing
4. **KC4:** Data Structures
5. **KC5:** Android Development
6. **KC6:** Tokeniser & Parser
7. **KC7:** Persistent Data
8. **KC8:** Refactoring
9. **KC9:** Design by Contract
10. **KC10:** Intellectual Property

---

## 🏗️ System Architecture

```text
Student Interaction History
[ (KC_i, Question_j, Correctness, Elapsed_Days), ... ]
                          │
                          ▼
            UKT Model (Self-Attention Backbone)
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
   P(correct) [Mastery]      Covariance Vector [Uncertainty]
            │                           │
            └─────────────┬─────────────┘
                          │
                          ▼
        Forgetting & Interference Risk Calculator
            │  - Logarithmic time-decay gap: log1p(Δt)
            │  - Pairwise concept interference risk matrix
            ▼
               Safe Mastery Computation
     Safe Mastery = P(correct) - λ_u * Uncertainty - λ_f * Forgetting
                          │
                          ▼
              Adaptive Policy Engine
       [ DIAGNOSE (🔵)  |  REVIEW (🟡)  |  ADVANCE (🟢) ]
                          │
                          ▼
        Curriculum-Aware Learning Path Planner
       [ Filter completed KCs → Recommended remedial path ]
```

---

## 📁 Repository Structure

```text
UKT_FORGET_SE_Web_Demo/
├── app.py                     # Main Streamlit web application (6 tabs)
├── setup_demo.py              # Environment verification script
├── requirements.txt           # Python dependencies
├── .gitignore                 # Standard exclusions (.venv, caches, etc.)
├── artifacts/                 # Pretrained weights and metadata (~10.5 MB)
│   ├── best.pt                # PyTorch UKT trained model checkpoint
│   ├── web_config.json        # Hyperparameters and architecture configs
│   ├── metrics.json           # Validation metrics (AUC, Accuracy, Brier)
│   ├── kc_mapping.json        # 10 KC definitions and metadata
│   ├── question_mapping.csv   # Mapping between raw questions and model IDs
│   ├── uncertainty_stats.json # Validation calibration quantiles (q05, q95)
│   ├── forgetting_stats.json  # Gap scaling and weighting configs
│   ├── interference_matrix.json # Inter-concept interference matrix
│   ├── policy_config.json     # Decision thresholds & penalty weights
│   ├── curriculum_graph.json  # Course sequence graph structure
│   └── demo_history.csv       # Sample student histories for testing
├── src/                       # Modular core application logic
│   ├── artifacts.py           # ArtifactStore for loading & validating assets
│   ├── model_service.py       # UKT model loader and PyTorch inference pipeline
│   ├── policy.py              # Safe mastery calculation & 3-tier action policy
│   ├── planner.py             # Curriculum-aware path planning algorithm
│   └── utils.py               # History generation & data normalization utilities
└── UKT/                       # Official UKT model architecture definitions
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python **3.10**, **3.11**, or **3.12**
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/<your-username>/UKT-FORGET-SE-Adaptive-Learning.git
cd UKT-FORGET-SE-Adaptive-Learning
```

### 3. Create & Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Verify Setup & Run Demo
```bash
python setup_demo.py
streamlit run app.py
```
The interactive application will automatically launch in your browser at `http://localhost:8501`.

---

## 🖥️ Web Demo Walkthrough

The demo includes **6 comprehensive tabs**:
1. **🏠 Giới thiệu & Cách dùng:** Overview of concepts, knowledge components, and instructions.
2. **🧑‍🎓 Lịch sử học tập:** Interactive data editor to customize, synthesize, or load de-identified student learning histories.
3. **🧠 Phân tích UKT:** Real-time inference to evaluate candidate questions, displaying $P(\text{correct})$, Uncertainty, Forgetting Risk, and prescribed Action (`DIAGNOSE` / `REVIEW` / `ADVANCE`).
4. **🛤️ Lộ trình:** Select a target KC to generate an optimized remedial learning roadmap.
5. **🗺️ Mapping & Dữ liệu:** Inspect dataset design, question mappings, and KC definitions.
6. **📊 Mô hình & Metrics:** Review validation performance, model hyperparameters, and checkpoint details.

---

## 📜 Citation & References

- **UKT:** Uncertainty-aware Knowledge Tracing ([GitHub](https://github.com/UncertaintyForKnowledgeTracing/UKT))
- **FORGET-SE:** An Empirical Study of Memory Decay and Interference in Computer Science Education.
