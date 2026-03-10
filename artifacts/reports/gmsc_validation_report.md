# Credit Risk Intelligence Engine — Model Validation Report

### GMSC Portfolio | Version 1.0 | Generated: March 2026

---

## 1. Executive Summary

This report documents the development, validation, and performance of the **Probability of Default (PD)** model built for the Give Me Some Credit (GMSC) consumer credit portfolio. The model is intended for use in **Basel III Expected Loss (EL) calculations**, credit decisioning support, and regulatory reporting.

| Key Metric | Value | Rating |
|---|---|---|
| **Best Model** | Random Forest | — |
| **AUC-ROC** | 0.8687 | ✅ Excellent |
| **Gini Coefficient** | 0.7375 | ✅ Excellent |
| **KS Statistic** | 0.5883 | ✅ Excellent |
| **Portfolio EL** | $46.6M | 13.85% of EAD |

**Conclusion:** The model demonstrates strong discriminatory power (AUC > 0.85, KS > 0.50), exceeding the regulatory minimum thresholds. It is recommended for deployment with the limitations noted in Section 7.

---

## 2. Model Development

### 2.1 Data Source

- **Dataset:** Give Me Some Credit (Kaggle)
- **Records:** 150,000 US consumer credit borrowers
- **Target:** `SeriousDlqin2yrs` — whether borrower experienced 90+ DPD within 2 years
- **Default Rate:** 6.7% (severe class imbalance)

### 2.2 Feature Engineering (Layer 1)

Features were engineered from raw credit bureau data using a repeatable PySpark + Pandas pipeline:

| Category | Features Created |
|---|---|
| **Delinquency Flags** | `flag_30dpd`, `flag_60dpd`, `flag_90dpd` — binary indicators |
| **Composite Scores** | `total_delinquency_score` — sum of all past-due events |
| **Log Transforms** | `log_MonthlyIncome`, `log_RevolvingUtilization`, `log_DebtRatio` |
| **Utilization Buckets** | Low / Medium / High / Critical credit utilization bands |
| **DTI Buckets** | Conservative / Typical / Stressed / Severe debt-to-income bands |
| **Age Cleaning** | Minimum age capped at 18 (legal lending age) |

### 2.3 Class Imbalance Treatment

Two complementary techniques were applied:

1. **SMOTE** (Synthetic Minority Oversampling) on the training set only
2. **class_weight='balanced'** in all model definitions as a secondary safeguard

### 2.4 Data Split

Stratified sampling maintaining the original default rate across all splits:

- **Training:** 70% (with SMOTE rebalancing)
- **Validation:** 15% (used for early stopping in boosting models)
- **Test:** 15% (final evaluation — never seen during training)

---

## 3. Model Selection

Four candidate models were trained and evaluated:

| Model | AUC-ROC | Gini | KS Statistic | F1 Score | Train Time |
|---|---|---|---|---|---|
| **Random Forest** 🏆 | **0.8687** | **0.7375** | **0.5883** | 0.3582 | 9.5s |
| Logistic Regression | 0.8644 | 0.7288 | 0.5873 | 0.3485 | 0.4s |
| XGBoost | 0.8655 | 0.7311 | 0.5803 | 0.3493 | 4.9s |
| LightGBM | 0.8621 | 0.7243 | 0.5791 | 0.3363 | 3.7s |

**Selection Criteria:** Random Forest was selected as the champion model based on the highest AUC-ROC (0.8687) on the held-out test set.

**Note on XGBoost/LightGBM:** Both boosting models now use **early stopping** (patience=30 rounds) with the validation set to prevent overfitting and reduce unnecessary training time.

---

## 4. Model Explainability (SHAP Analysis)

SHAP (SHapley Additive exPlanations) values were computed for full regulatory transparency:

### 4.1 Top 5 Features by SHAP Importance

| Rank | Feature | Mean |SHAP| | Business Interpretation |
|---|---|---|---|
| 1 | `RevolvingUtilizationOfUnsecuredLines` | 0.0818 | High utilization = maxed-out credit = highest default risk |
| 2 | `log_RevolvingUtilizationOfUnsecuredLines` | 0.0725 | Log-transform captures extreme utilization tail |
| 3 | `total_delinquency_score` | 0.0711 | Composite: more past-due events = higher risk |
| 4 | `age` | 0.0290 | Younger borrowers carry higher default risk |
| 5 | `flag_90dpd` | 0.0257 | Any 90+ DPD history is a strong default predictor |

### 4.2 Explainability Artifacts

- **Beeswarm Plot:** `artifacts/plots/gmsc_shap_beeswarm.png`
- **Bar Chart:** `artifacts/plots/gmsc_shap_bar.png`
- **Waterfall (Single Loan):** `artifacts/plots/gmsc_shap_waterfall.png`

---

## 5. Basel III Expected Loss

Expected Loss was computed using the standard Basel formula:

> **EL = PD × LGD × EAD**

### 5.1 Assumptions

| Parameter | Value | Source |
|---|---|---|
| PD | Model-predicted | Random Forest `predict_proba()` |
| LGD | 0.45 (45%) | Basel III standard for unsecured consumer credit |
| EAD | $15,000 (fixed) | Assumed average exposure per borrower |

### 5.2 Portfolio Results

| Metric | Value |
|---|---|
| Total Loans Evaluated | 22,409 |
| Total Exposure (EAD) | $336,135,000 |
| Total Expected Loss | $46,569,523 |
| EL as % of EAD | 13.85% |
| Average PD | 0.3079 |
| Median PD | 0.2230 |

### 5.3 Risk Grade Segmentation

Loans were segmented into Basel-style risk grades based on predicted PD:

| Grade | PD Range | Risk Level |
|---|---|---|
| A | 0% – 5% | Very Low |
| B | 5% – 10% | Low |
| C | 10% – 20% | Medium |
| D | 20% – 50% | High |
| E | 50% – 100% | Very High |

Detailed segment data: `artifacts/metrics/gmsc_el_by_segment.csv`

---

## 6. Deep Learning Layer (Layer 3)

### 6.1 Autoencoder Anomaly Detection

- **Architecture:** PyTorch Autoencoder (37 → 32 → 16 → 8 → 16 → 32 → 37)
- **Training:** On healthy (non-default) loans only
- **Scoring:** Reconstruction error as anomaly signal
- **Ensemble:** Combined with Isolation Forest for rule-based + deep-learning anomaly flags
- **Result:** Anomalous loans default at a significantly higher rate than normal loans

### 6.2 LSTM Sequential Model

- **Architecture:** 2-layer LSTM (hidden_dim=64) + classifier head
- **Purpose:** Detect credit deterioration velocity over time
- **Input Shape:** [batch, 3 time steps, 36 features]
- **Status:** Trained and functional with simulated temporal sequences

---

## 7. Limitations & Risks

| # | Limitation | Mitigation |
|---|---|---|
| 1 | Trained on 2011 US data — may not generalize to other periods/geographies | Retrain on recent data before production deployment |
| 2 | LGD and EAD are constants — real values vary by loan | Integrate loan-level LGD/EAD from servicing data |
| 3 | Low precision (23.4%) — model over-flags non-defaults | Acceptable tradeoff: missed defaults ($) >> false alarms |
| 4 | No macroeconomic stress testing in v1.0 | Layer 4 (planned) adds GDP/unemployment shock scenarios |
| 5 | LSTM uses simulated temporal data | Requires Freddie Mac monthly performance logs for production |

---

## 8. Regulatory Compliance

| Requirement | Status |
|---|---|
| SR 11-7 Model Risk Management | ✅ Model card, validation report, SHAP explanations provided |
| Fair Lending (ECOA / Reg B) | ✅ No protected class variables used |
| Basel III Capital Requirements | ✅ EL computed with standard PD × LGD × EAD formula |
| Model Documentation | ✅ WHY/HOW/WHAT docs for each layer |
| Reproducibility | ✅ Fixed random seed (42), version-controlled pipeline |

---

## 9. Artifacts & File Index

| Artifact | Path |
|---|---|
| Trained Model | `models/gmsc_pd_model.pkl` |
| Feature Scaler | `models/gmsc_scaler.pkl` |
| Feature Names | `models/gmsc_feature_names.json` |
| Model Metrics | `artifacts/metrics/gmsc_model_metrics.json` |
| EL Summary | `artifacts/metrics/gmsc_el_summary.json` |
| EL Segments | `artifacts/metrics/gmsc_el_by_segment.csv` |
| SHAP Importance | `artifacts/metrics/gmsc_shap_importance.json` |
| Training Log | `artifacts/metrics/gmsc_training_log.txt` |
| ROC Curves | `artifacts/plots/gmsc_roc_curves.png` |
| PR Curves | `artifacts/plots/gmsc_pr_curves.png` |
| KS Curve | `artifacts/plots/gmsc_ks_curve.png` |
| Confusion Matrix | `artifacts/plots/gmsc_confusion_matrix.png` |
| PD Distribution | `artifacts/plots/gmsc_pd_distribution.png` |
| Model Card | `artifacts/model_cards/gmsc_model_card.json` |
| Dashboard | `dashboard/app.py` |

---

*Report generated by the Risk Intelligence Engine automated pipeline.*
*For questions, contact the Model Validation team.*
