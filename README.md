<p align="center">
  <h1 align="center">🏦 Risk Intelligence Engine</h1>
  <p align="center">
    <strong>End-to-end Credit Risk ML Platform — Basel III Compliant</strong>
  </p>
  <p align="center">
    <a href="#-quick-start"><img src="https://img.shields.io/badge/Quick_Start-▶-6C63FF?style=for-the-badge" alt="Quick Start"/></a>
    <a href="#-architecture"><img src="https://img.shields.io/badge/Architecture-📐-00D4AA?style=for-the-badge" alt="Architecture"/></a>
    <a href="#-results"><img src="https://img.shields.io/badge/Results-📊-FF6B6B?style=for-the-badge" alt="Results"/></a>
    <a href="#-dashboard"><img src="https://img.shields.io/badge/Dashboard-🖥-FFD93D?style=for-the-badge" alt="Dashboard"/></a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python"/>
    <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"/>
    <img src="https://img.shields.io/badge/PySpark-3.5-E25A1C?logo=apachespark&logoColor=white" alt="PySpark"/>
    <img src="https://img.shields.io/badge/Streamlit-1.30-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"/>
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
  </p>
</p>

---

A production-grade **credit risk analytics platform** that ingests raw loan data, engineers features at scale with PySpark, trains an ensemble of ML models, applies deep learning anomaly detection, computes Basel III Expected Loss, and serves everything through an interactive Streamlit dashboard — all with **one command**.

## ✨ Key Highlights

| Feature | Details |
|---|---|
| 🎯 **AUC-ROC: 0.8687** | Random Forest champion model — industry-grade discriminatory power |
| 📊 **KS Statistic: 0.5883** | Excellent separation between defaults and non-defaults |
| 🏛️ **Basel III Compliant** | Full EL = PD × LGD × EAD computation with risk grade segmentation |
| 🧠 **Deep Learning** | PyTorch Autoencoder for anomaly detection + LSTM for credit deterioration |
| 🔍 **SHAP Explainability** | Per-prediction explanations meeting SR 11-7 regulatory requirements |
| 🚀 **One-Command Pipeline** | `python run_full_pipeline.py` generates everything automatically |

---

## 📐 Architecture

```
                        ┌──────────────────────────────────────────────┐
                        │         Risk Intelligence Engine             │
                        └──────────────────────────────────────────────┘
                                          │
          ┌───────────────────────────────┼───────────────────────────────┐
          │                               │                               │
    ┌─────▼─────┐                  ┌──────▼──────┐                ┌──────▼──────┐
    │  Layer 1   │                  │   Layer 2    │                │   Layer 3    │
    │   Data     │                  │   ML Models  │                │ Deep Learning│
    │Engineering │                  │  PD Scoring  │                │  Anomaly +   │
    │            │                  │              │                │    LSTM      │
    │ PySpark    │──── Parquet ────▶│ LR / RF /    │── Best Model──▶│ Autoencoder  │
    │ Ingestion  │   Feature Store  │ XGB / LGBM   │                │ Isolation    │
    │ Feature    │                  │ SMOTE        │                │ Forest       │
    │ Engineering│                  │ Early Stop   │                │ PyTorch      │
    └────────────┘                  └──────┬───────┘                └──────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
              ┌─────▼─────┐        ┌──────▼──────┐       ┌──────▼──────┐
              │   SHAP     │        │  Basel III   │       │  Dashboard  │
              │Explainabilit│        │Expected Loss │       │  Streamlit  │
              │Beeswarm    │        │ EL=PD×LGD×EAD│       │  + Plotly   │
              │Waterfall   │        │ Risk Grades  │       │  7 Pages    │
              └────────────┘        └──────────────┘       └─────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.10+
Java 8+ (for PySpark)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/ARYANRAJ1121/risk-intelligence-engine.git
cd risk-intelligence-engine

# Install dependencies
pip install -r requirements.txt
```

### Download Data

Place raw CSV files in the `data/raw/` directory:

| Dataset | Source | File |
|---|---|---|
| **GMSC** | [Kaggle — Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit) | `cs-training.csv` |
| **LendingClub** | [Kaggle — Lending Club](https://www.kaggle.com/datasets/wordsforthewise/lending-club) | `accepted_2007_to_2018Q4.csv` |

### Run the Full Pipeline

```bash
# Process GMSC dataset (recommended first run — ~30 seconds)
python run_full_pipeline.py --dataset gmsc

# Process LendingClub dataset
python run_full_pipeline.py --dataset lc

# Process both datasets
python run_full_pipeline.py --dataset all

# Force rebuild feature store from scratch
python run_full_pipeline.py --dataset gmsc --force-layer1
```

### Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

Open **<http://localhost:8501>** in your browser.

---

## 📊 Results

### Model Performance (GMSC Dataset)

| Model | AUC-ROC | Gini | KS Statistic | F1 Score | Train Time |
|---|---|---|---|---|---|
| **Random Forest** 🏆 | **0.8687** | **0.7375** | **0.5883** | 0.3582 | 9.5s |
| Logistic Regression | 0.8644 | 0.7288 | 0.5873 | 0.3485 | 0.4s |
| XGBoost | 0.8655 | 0.7311 | 0.5803 | 0.3493 | 4.9s |
| LightGBM | 0.8621 | 0.7243 | 0.5791 | 0.3363 | 3.7s |

### Basel III Expected Loss

| Metric | Value |
|---|---|
| Total Loans Evaluated | 22,409 |
| Total Exposure (EAD) | $336,135,000 |
| **Total Expected Loss** | **$46,569,523** |
| EL as % of EAD | 13.85% |
| Average PD | 0.3079 |

### Top 5 Risk Drivers (SHAP)

| Rank | Feature | Mean \|SHAP\| | Why It Matters |
|---|---|---|---|
| 1 | `RevolvingUtilizationOfUnsecuredLines` | 0.0818 | Maxed-out credit = highest default signal |
| 2 | `log_RevolvingUtilization` | 0.0725 | Captures extreme utilization tail |
| 3 | `total_delinquency_score` | 0.0711 | More past-due events → higher risk |
| 4 | `age` | 0.0290 | Younger borrowers = higher default rate |
| 5 | `flag_90dpd` | 0.0257 | Any 90+ DPD history is a red flag |

---

## 🖥 Dashboard

The interactive Streamlit dashboard has **7 pages**:

| Page | Description |
|---|---|
| **📊 Command Center** | KPI cards, animated gauges (AUC, Gini, KS, EL/EAD), model comparison bar chart, radar chart |
| **🤖 Model Arena** | ROC curves, Precision-Recall curves, KS statistic plot, confusion matrix, PD distribution |
| **🔍 Explainability** | Interactive SHAP bar chart, beeswarm plot, waterfall plot, top features table |
| **⚠️ Anomaly Radar** | Autoencoder reconstruction error distribution, ensemble anomaly methodology |
| **📋 Risk Grid** | Basel III risk grade donut chart, EL waterfall by grade, segment data table |
| **🔬 Data Explorer** | Interactive feature distribution histograms, raw data preview |
| **📄 Governance** | Model card, intended use, limitations, ethical considerations, full validation report |

---

## 📁 Project Structure

```
risk-intelligence-engine/
│
├── run_full_pipeline.py          # 🚀 ONE COMMAND — runs everything
│
├── src/
│   ├── ingestion/                # Layer 1: PySpark data pipeline
│   │   ├── config.py             #   Centralized configuration
│   │   ├── ingest.py             #   PySpark CSV ingestion + validation
│   │   ├── run_pipeline.py       #   Layer 1 orchestrator
│   │   └── db_writer.py          #   Parquet feature store writer
│   │
│   ├── features/                 # Feature Engineering
│   │   ├── feature_engineer.py   #   GMSC + LendingClub feature pipelines
│   │   └── macro_enrichment.py   #   FRED macroeconomic data join
│   │
│   ├── models/                   # Layer 2 + 3: ML & DL Models
│   │   ├── train_pd_model.py     #   4-model tournament (LR, RF, XGB, LGBM)
│   │   ├── evaluate.py           #   Metrics, ROC, PR, KS, confusion matrix
│   │   ├── dl_anomaly.py         #   PyTorch Autoencoder + Isolation Forest
│   │   ├── lstm_sequence.py      #   LSTM credit deterioration detector
│   │   └── train_dl.py           #   Deep learning training orchestrator
│   │
│   ├── explainability/           # SHAP Analysis
│   │   └── shap_explainer.py     #   Beeswarm, waterfall, bar plots
│   │
│   └── risk_engine/              # Basel III Calculations
│       └── expected_loss.py      #   EL = PD × LGD × EAD + risk grading
│
├── dashboard/
│   └── app.py                    # Streamlit dashboard (7 pages, Plotly)
│
├── artifacts/
│   ├── metrics/                  # JSON/CSV model performance data
│   ├── plots/                    # PNG evaluation visualizations
│   ├── model_cards/              # Model governance documentation
│   └── reports/                  # Auto-generated validation reports
│
├── models/                       # Serialized model files (.pkl)
├── tests/                        # Unit tests (37 passing)
├── data/
│   ├── raw/                      # Raw CSV files (not tracked)
│   └── feature_store/            # Parquet feature store
│
├── reports/                      # Project-level summary
├── requirements.txt              # Python dependencies
└── README.md                     # You are here
```

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Expected output: 37 passed
```

---

## 🔧 Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Data Engineering** | PySpark 3.5, Pandas | Large-scale CSV ingestion, validation, transformation |
| **Feature Store** | Apache Parquet | Columnar storage for fast analytical queries |
| **ML Models** | scikit-learn, XGBoost, LightGBM | 4-model PD classification tournament |
| **Deep Learning** | PyTorch | Autoencoder anomaly detection, LSTM sequences |
| **Explainability** | SHAP | Per-prediction feature contribution analysis |
| **Visualization** | Matplotlib, Plotly | Static plots + interactive dashboard charts |
| **Dashboard** | Streamlit | 7-page interactive web application |
| **Class Imbalance** | imbalanced-learn (SMOTE) | Synthetic oversampling of minority class |

---

## 📋 Regulatory Compliance

| Standard | Implementation |
|---|---|
| **Basel III / IV** | EL = PD × LGD × EAD computed per borrower with risk grade segmentation (A–E) |
| **SR 11-7** (Fed Model Risk Mgmt) | Model card, validation report, SHAP explanations, limitations documented |
| **ECOA / Reg B** (Fair Lending) | No protected class variables (race, gender, religion) used as features |
| **GDPR Art. 22** (Right to Explanation) | SHAP waterfall provides per-prediction explanations |

---

## 🗺 Roadmap

- [x] Layer 1: PySpark data engineering pipeline
- [x] Layer 2: 4-model PD tournament with SMOTE + early stopping
- [x] Layer 3: PyTorch Autoencoder + LSTM
- [x] Layer 5: Streamlit dashboard with Plotly interactive charts
- [x] Auto-generation: Model cards + validation reports
- [x] One-command automation pipeline
- [ ] Layer 4: Macroeconomic stress testing (GDP/unemployment shocks)
- [ ] API endpoint for real-time loan scoring
- [ ] Model monitoring and drift detection
- [ ] Docker containerization

---

## 👤 Author

**Aryan Raj**

- GitHub: [@ARYANRAJ1121](https://github.com/ARYANRAJ1121)

---

## 📄 License

This project is licensed under the MIT License.
