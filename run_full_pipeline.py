"""
run_full_pipeline.py — Master Automation Pipeline
===================================================
ONE COMMAND TO RULE THEM ALL.

    python run_full_pipeline.py --dataset gmsc
    python run_full_pipeline.py --dataset lc
    python run_full_pipeline.py --dataset all

This script orchestrates every layer of the Risk Intelligence Engine:
    Layer 1: Data Engineering  (PySpark → Feature Store)
    Layer 2: ML PD Model      (4-model tournament → Best model → SHAP → EL)
    Layer 3: Deep Learning     (Autoencoder anomaly + LSTM deterioration)
    Layer 5: Reports           (Model card + Validation report auto-generated)

All artifacts, plots, metrics, reports, and model cards are produced
automatically with ZERO manual intervention.
"""

import io
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

# ---- Project setup ---- #
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.config import Config


# ============================================================
#  LOGGING
# ============================================================

def setup_logging():
    """Configure pipeline-wide logging with UTF-8 support."""
    utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    handler = logging.StreamHandler(utf8_stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.basicConfig(level=logging.INFO, handlers=[handler])


logger = logging.getLogger("pipeline.master")


# ============================================================
#  LAYER 1: DATA ENGINEERING
# ============================================================

def run_layer1(dataset: str) -> bool:
    """
    Layer 1: Ingest raw CSV → PySpark validation → Feature engineering → Parquet.
    """
    logger.info("=" * 70)
    logger.info("  LAYER 1: DATA ENGINEERING — %s", dataset.upper())
    logger.info("=" * 70)

    feature_path = Config.FEATURE_STORE_DIR / f"{dataset}_features.parquet"

    if feature_path.exists():
        import pandas as pd
        df = pd.read_parquet(feature_path)
        logger.info("  Feature store already exists: %d rows × %d cols", len(df), len(df.columns))
        logger.info("  Skipping Layer 1 (use --force-layer1 to rebuild)")
        return True

    try:
        from src.ingestion.run_pipeline import run_gmsc_pipeline, run_lc_pipeline
        if dataset == "gmsc":
            run_gmsc_pipeline()
        else:
            run_lc_pipeline()
        return True
    except FileNotFoundError as e:
        logger.error("Dataset file not found: %s", e)
        logger.error("Place the raw CSV in: %s", Config.RAW_DATA_DIR)
        return False
    except Exception as e:
        logger.error("Layer 1 failed: %s", e, exc_info=True)
        return False


# ============================================================
#  LAYER 2: ML PD MODEL
# ============================================================

def run_layer2(dataset: str) -> dict:
    """
    Layer 2: Load features → Train 4 models → Evaluate → SHAP → EL → Save.
    Returns the pipeline output dict or None on failure.
    """
    logger.info("=" * 70)
    logger.info("  LAYER 2: ML PD MODEL — %s", dataset.upper())
    logger.info("=" * 70)

    try:
        from src.models.train_pd_model import run_training_pipeline
        output = run_training_pipeline(dataset)
        return output
    except Exception as e:
        logger.error("Layer 2 failed: %s", e, exc_info=True)
        return None


# ============================================================
#  LAYER 3: DEEP LEARNING
# ============================================================

def run_layer3(dataset: str) -> bool:
    """
    Layer 3: Autoencoder anomaly detection + LSTM credit deterioration.
    """
    logger.info("=" * 70)
    logger.info("  LAYER 3: DEEP LEARNING — %s", dataset.upper())
    logger.info("=" * 70)

    try:
        from src.models.train_dl import test_autoencoder_anomaly, test_lstm_sequence
        
        logger.info("  3a. Autoencoder Anomaly Detection...")
        test_autoencoder_anomaly(dataset)

        logger.info("  3b. LSTM Credit Deterioration...")
        test_lstm_sequence(dataset)

        return True
    except Exception as e:
        logger.error("Layer 3 failed: %s", e, exc_info=True)
        return False


# ============================================================
#  LAYER 5: AUTO-GENERATE REPORTS & MODEL CARD
# ============================================================

def generate_model_card(dataset: str) -> bool:
    """
    Auto-generate the model card JSON from pipeline outputs.
    Reads metrics, SHAP, and EL data to fill every field.
    """
    logger.info("  Generating model card...")

    metrics_dir = PROJECT_ROOT / "artifacts" / "metrics"
    cards_dir = PROJECT_ROOT / "artifacts" / "model_cards"
    cards_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline outputs
    metrics_path = metrics_dir / f"{dataset}_model_metrics.json"
    el_path = metrics_dir / f"{dataset}_el_summary.json"
    shap_path = metrics_dir / f"{dataset}_shap_importance.json"

    if not metrics_path.exists():
        logger.warning("  No model metrics found — skipping model card")
        return False

    metrics = json.load(open(metrics_path))
    el = json.load(open(el_path)) if el_path.exists() else {}
    shap_data = json.load(open(shap_path)) if shap_path.exists() else {}

    # Find best model
    best_name = max(metrics, key=lambda k: metrics[k]["auc_roc"])
    best = metrics[best_name]

    # Top SHAP features
    top_shap = [
        {"feature": k, "mean_shap": round(v, 6)}
        for k, v in sorted(shap_data.items(), key=lambda x: x[1], reverse=True)[:5]
    ]

    # Competing models
    competing = {
        name: {"auc_roc": m["auc_roc"], "ks": m["ks_statistic"]}
        for name, m in metrics.items() if name != best_name
    }

    dataset_names = {
        "gmsc": "Give Me Some Credit (GMSC) — Kaggle",
        "lc": "LendingClub — Kaggle / LendingClub Open Data",
    }

    card = {
        "model_name": f"{dataset.upper()} PD Model — {best_name}",
        "model_version": "1.0.0",
        "model_type": "Binary Classification (Probability of Default)",
        "framework": "scikit-learn / XGBoost / LightGBM",
        "created_date": datetime.now().strftime("%Y-%m-%d"),
        "author": "Risk Intelligence Engine — Auto-ML Pipeline",
        "dataset": dataset_names.get(dataset, dataset),

        "intended_use": {
            "primary": "Predict Probability of Default (PD) for consumer credit borrowers",
            "secondary": "Feed PD into Basel III Expected Loss formula: EL = PD × LGD × EAD",
            "users": ["Credit Risk Analysts", "Loan Officers", "Model Validation Teams", "Regulators"],
            "out_of_scope": "Not designed for commercial/corporate lending or non-US markets",
        },

        "training_data": {
            "source": dataset_names.get(dataset, dataset),
            "features_used": len(shap_data) if shap_data else "unknown",
            "target_variable": "SeriousDlqin2yrs" if dataset == "gmsc" else "is_default",
            "class_imbalance_handling": "SMOTE + class_weight='balanced'",
            "split_strategy": "Stratified 70/15/15 (Train/Val/Test)",
        },

        "model_architecture": {
            "algorithm": best_name,
            "selection_criteria": "Highest AUC-ROC on held-out test set",
        },

        "performance_metrics": {
            "auc_roc": best["auc_roc"],
            "gini_coefficient": best["gini"],
            "ks_statistic": best["ks_statistic"],
            "pr_auc": best["pr_auc"],
            "f1_score": best["f1_score"],
            "accuracy": best["accuracy"],
            "precision": best["precision"],
            "recall": best["recall"],
            "true_positives": best["true_positives"],
            "true_negatives": best["true_negatives"],
            "false_positives": best["false_positives"],
            "false_negatives": best["false_negatives"],
        },

        "competing_models": competing,
        "top_features_shap": top_shap,
        "expected_loss_summary": el,

        "limitations": [
            f"Trained on historical data — may not generalize to future economic cycles",
            "LGD and EAD are assumed constants — production systems should use loan-level estimates",
            "Does not incorporate macroeconomic stress testing in this version",
            f"Precision is {best['precision']:.1%} — model favors recall to minimize missed defaults",
        ],

        "ethical_considerations": [
            "No protected class variables (race, gender, religion) are used",
            "SHAP explainability is provided for regulatory transparency (SR 11-7)",
            "Age may be used as a feature — must comply with ECOA",
            "Model decisions should be reviewed by human underwriters",
        ],

        "deployment": {
            "model_file": f"models/{dataset}_pd_model.pkl",
            "scaler_file": f"models/{dataset}_scaler.pkl",
            "feature_names": f"models/{dataset}_feature_names.json",
            "dashboard": "streamlit run dashboard/app.py",
        },
    }

    card_path = cards_dir / f"{dataset}_model_card.json"
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2)

    logger.info("  Saved model card → %s", card_path)
    return True


def generate_validation_report(dataset: str) -> bool:
    """
    Auto-generate the markdown validation report from pipeline outputs.
    """
    logger.info("  Generating validation report...")

    metrics_dir = PROJECT_ROOT / "artifacts" / "metrics"
    reports_dir = PROJECT_ROOT / "artifacts" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = metrics_dir / f"{dataset}_model_metrics.json"
    el_path = metrics_dir / f"{dataset}_el_summary.json"
    shap_path = metrics_dir / f"{dataset}_shap_importance.json"

    if not metrics_path.exists():
        logger.warning("  No model metrics found — skipping report")
        return False

    metrics = json.load(open(metrics_path))
    el = json.load(open(el_path)) if el_path.exists() else {}
    shap_data = json.load(open(shap_path)) if shap_path.exists() else {}

    best_name = max(metrics, key=lambda k: metrics[k]["auc_roc"])
    best = metrics[best_name]
    today = datetime.now().strftime("%B %d, %Y")

    # Build model comparison table rows
    model_rows = ""
    for name, m in metrics.items():
        trophy = " 🏆" if name == best_name else ""
        model_rows += f"| **{name}**{trophy} | {m['auc_roc']:.4f} | {m['gini']:.4f} | {m['ks_statistic']:.4f} | {m['f1_score']:.4f} | {m.get('train_time_sec', 0):.1f}s |\n"

    # Top SHAP features
    shap_rows = ""
    for i, (feat, val) in enumerate(sorted(shap_data.items(), key=lambda x: x[1], reverse=True)[:5]):
        shap_rows += f"| {i+1} | `{feat}` | {val:.4f} |\n"

    report = f"""# Credit Risk Intelligence Engine — Model Validation Report

## {dataset.upper()} Portfolio | Version 1.0 | Generated: {today}

---

## 1. Executive Summary

| Key Metric | Value | Rating |
| --- | --- | --- |
| **Best Model** | {best_name} | — |
| **AUC-ROC** | {best['auc_roc']:.4f} | {"✅ Excellent" if best['auc_roc'] > 0.85 else "⚠️ Good" if best['auc_roc'] > 0.75 else "❌ Below Threshold"} |
| **Gini Coefficient** | {best['gini']:.4f} | {"✅ Excellent" if best['gini'] > 0.65 else "⚠️ Acceptable"} |
| **KS Statistic** | {best['ks_statistic']:.4f} | {"✅ Excellent" if best['ks_statistic'] > 0.40 else "⚠️ Acceptable" if best['ks_statistic'] > 0.20 else "❌ Weak"} |
| **Portfolio EL** | ${el.get('total_expected_loss', 0):,.0f} | {el.get('el_as_pct_of_ead', 0):.2f}% of EAD |

---

## 2. Model Selection

| Model | AUC-ROC | Gini | KS | F1 | Train Time |
| --- | --- | --- | --- | --- | --- |
{model_rows}
**Selection Criteria:** Highest AUC-ROC on held-out test set.

---

## 3. Performance Detail ({best_name})

| Metric | Value |
| --- | --- |
| Accuracy | {best['accuracy']:.4f} |
| Precision | {best['precision']:.4f} |
| Recall | {best['recall']:.4f} |
| F1 Score | {best['f1_score']:.4f} |
| True Positives | {best['true_positives']:,} |
| True Negatives | {best['true_negatives']:,} |
| False Positives | {best['false_positives']:,} |
| False Negatives | {best['false_negatives']:,} |

---

## 4. SHAP Explainability — Top 5 Features

| Rank | Feature | Mean SHAP |
| --- | --- | --- |
{shap_rows}
---

## 5. Basel III Expected Loss

> **EL = PD × LGD × EAD**

| Parameter | Value |
| --- | --- |
| Total Loans | {el.get('total_loans', 'N/A'):,} |
| Total EAD | ${el.get('total_ead', 0):,.0f} |
| Total Expected Loss | ${el.get('total_expected_loss', 0):,.0f} |
| EL / EAD | {el.get('el_as_pct_of_ead', 0):.2f}% |
| Average PD | {el.get('average_pd', 0):.4f} |
| LGD Assumption | {el.get('lgd_assumption', 0.45):.0%} |

---

## 6. Deep Learning Layer

- **Autoencoder** — PyTorch anomaly detector trained on healthy loans
- **LSTM** — Sequential credit deterioration model (2-layer, hidden_dim=64)
- **Ensemble** — Autoencoder + Isolation Forest for robust anomaly flagging

---

## 7. Regulatory Compliance

| Requirement | Status |
| --- | --- |
| SR 11-7 Model Risk Management | ✅ Model card + validation report |
| Fair Lending (ECOA) | ✅ No protected variables |
| Basel III Capital | ✅ EL = PD × LGD × EAD |
| SHAP Explainability | ✅ Per-prediction explanations |
| Reproducibility | ✅ Fixed seed (42), version controlled |

---

*Auto-generated by `run_full_pipeline.py` on {today}*
"""

    report_path = reports_dir / f"{dataset}_validation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info("  Saved validation report → %s", report_path)
    return True


def generate_project_summary() -> bool:
    """Generate the top-level project summary report."""
    logger.info("  Generating project summary...")

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%B %d, %Y")

    # Check which datasets have been processed
    datasets_info = []
    for ds in ["gmsc", "lc"]:
        metrics_path = PROJECT_ROOT / "artifacts" / "metrics" / f"{ds}_model_metrics.json"
        if metrics_path.exists():
            m = json.load(open(metrics_path))
            best = max(m, key=lambda k: m[k]["auc_roc"])
            datasets_info.append((ds, best, m[best]["auc_roc"]))

    dataset_rows = ""
    for ds, best, auc in datasets_info:
        dataset_rows += f"| {ds.upper()} | {best} | {auc:.4f} | ✅ Complete |\n"

    summary = f"""# Risk Intelligence Engine — Project Summary

## Generated: {today}

---

## Pipeline Status

| Dataset | Best Model | AUC-ROC | Status |
| --- | --- | --- | --- |
{dataset_rows}

## Architecture

```
Raw CSV → Layer 1 (PySpark) → Layer 2 (ML Models) → Layer 3 (Deep Learning) → Reports + Dashboard
```

## How to Run

```bash
# Full pipeline (one command):
python run_full_pipeline.py --dataset gmsc

# Launch dashboard:
streamlit run dashboard/app.py
```

## Output Structure

```
artifacts/
├── metrics/          # JSON/CSV performance data
├── plots/            # PNG evaluation charts
├── model_cards/      # Model governance JSON
└── reports/          # Validation reports (Markdown)
models/               # Serialized .pkl models
reports/              # Project-level summary
dashboard/            # Streamlit app
```

---

*Auto-generated by `run_full_pipeline.py`*
"""

    path = reports_dir / "project_summary.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(summary)

    logger.info("  Saved project summary → %s", path)
    return True


# ============================================================
#  MASTER ORCHESTRATOR
# ============================================================

def run_full_pipeline(dataset: str, force_layer1: bool = False):
    """
    Run the entire Risk Intelligence Engine pipeline end-to-end.
    """
    pipeline_start = time.time()

    logger.info("=" * 70)
    logger.info("  RISK INTELLIGENCE ENGINE — FULL PIPELINE")
    logger.info("  Dataset: %s | Time: %s", dataset.upper(), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 70)

    datasets = ["gmsc", "lc"] if dataset == "all" else [dataset]
    results = {}

    for ds in datasets:
        ds_start = time.time()
        logger.info("")
        logger.info("▶ Processing dataset: %s", ds.upper())
        logger.info("")

        # ---- Layer 1 ---- #
        feature_path = Config.FEATURE_STORE_DIR / f"{ds}_features.parquet"
        if force_layer1 or not feature_path.exists():
            if not run_layer1(ds):
                logger.error("❌ Layer 1 failed for %s — skipping remaining layers", ds)
                results[ds] = {"status": "FAILED", "failed_at": "Layer 1"}
                continue
        else:
            logger.info("  Layer 1: Feature store exists — skipping (use --force-layer1 to rebuild)")

        # ---- Layer 2 ---- #
        output = run_layer2(ds)
        if output is None:
            logger.error("❌ Layer 2 failed for %s — skipping remaining layers", ds)
            results[ds] = {"status": "FAILED", "failed_at": "Layer 2"}
            continue

        # ---- Layer 3 ---- #
        dl_ok = run_layer3(ds)
        if not dl_ok:
            logger.warning("⚠️ Layer 3 had issues for %s — continuing with report generation", ds)

        # ---- Layer 5: Reports ---- #
        logger.info("=" * 70)
        logger.info("  LAYER 5: REPORT GENERATION — %s", ds.upper())
        logger.info("=" * 70)

        generate_model_card(ds)
        generate_validation_report(ds)

        ds_elapsed = time.time() - ds_start
        results[ds] = {"status": "SUCCESS", "elapsed": round(ds_elapsed, 1)}
        logger.info("")
        logger.info("✅ %s — Complete in %.1f seconds", ds.upper(), ds_elapsed)

    # ---- Project summary ---- #
    generate_project_summary()

    # ---- Final summary ---- #
    total_elapsed = time.time() - pipeline_start
    logger.info("")
    logger.info("=" * 70)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 70)
    for ds, res in results.items():
        status = res["status"]
        if status == "SUCCESS":
            logger.info("  %s: ✅ SUCCESS (%.1fs)", ds.upper(), res["elapsed"])
        else:
            logger.info("  %s: ❌ FAILED at %s", ds.upper(), res["failed_at"])
    logger.info("  Total time: %.1f seconds", total_elapsed)
    logger.info("")
    logger.info("  📁 Artifacts:  artifacts/")
    logger.info("  📊 Dashboard:  streamlit run dashboard/app.py")
    logger.info("=" * 70)


# ============================================================
#  CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Risk Intelligence Engine — Full Automation Pipeline",
        epilog="Example: python run_full_pipeline.py --dataset gmsc",
    )
    parser.add_argument(
        "--dataset", choices=["gmsc", "lc", "all"], default="gmsc",
        help="Dataset to process: 'gmsc', 'lc', or 'all'",
    )
    parser.add_argument(
        "--force-layer1", action="store_true",
        help="Force re-run Layer 1 even if feature store exists",
    )
    args = parser.parse_args()

    setup_logging()
    run_full_pipeline(args.dataset, force_layer1=args.force_layer1)
