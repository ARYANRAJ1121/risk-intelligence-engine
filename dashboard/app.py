"""
app.py — Credit Risk Intelligence Dashboard
=============================================
WHY THIS EXISTS:
    Bank risk committees, loan officers, and regulators need to
    visualize model outputs, review portfolio risk, and drill into
    individual loans. This Streamlit app is the single pane of glass
    for the entire Risk Intelligence Engine.

HOW TO RUN:
    streamlit run dashboard/app.py
"""

import sys
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---- Project paths ---- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
MODELS_DIR = PROJECT_ROOT / "models"

# ============================================================
#  PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Risk Intelligence Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
#  CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        font-size: 1rem;
        opacity: 0.8;
        margin: 0.3rem 0 0 0;
    }

    .metric-card {
        background: linear-gradient(135deg, #1e1e2f, #2a2a40);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #6C63FF;
    }
    .metric-label {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.6);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }

    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #6C63FF;
        display: inline-block;
    }

    .risk-grade-A { color: #4CAF50; font-weight: 700; }
    .risk-grade-B { color: #8BC34A; font-weight: 700; }
    .risk-grade-C { color: #FFC107; font-weight: 700; }
    .risk-grade-D { color: #FF5722; font-weight: 700; }
    .risk-grade-E { color: #B71C1C; font-weight: 700; }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #1a1a2e);
    }
    div[data-testid="stSidebar"] .stRadio label {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
#  DATA LOADING HELPERS
# ============================================================

@st.cache_data
def load_metrics(dataset: str) -> dict:
    path = METRICS_DIR / f"{dataset}_model_metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

@st.cache_data
def load_el_summary(dataset: str) -> dict:
    path = METRICS_DIR / f"{dataset}_el_summary.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

@st.cache_data
def load_el_segments(dataset: str) -> pd.DataFrame:
    path = METRICS_DIR / f"{dataset}_el_by_segment.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_shap_importance(dataset: str) -> dict:
    path = METRICS_DIR / f"{dataset}_shap_importance.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ============================================================
#  SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🏦 Navigation")
    page = st.radio(
        "Select View",
        ["📊 Portfolio Overview", "🤖 Model Performance",
         "🔍 SHAP Explainability", "⚠️ Anomaly Detection",
         "📋 Risk Segments"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    dataset = st.selectbox("Dataset", ["gmsc", "lc"], format_func=lambda x: "GMSC (150K loans)" if x == "gmsc" else "LendingClub (1.3M loans)")

    st.markdown("---")
    st.markdown("### 📁 Quick Links")
    st.markdown(f"**Plots:** `artifacts/plots/`")
    st.markdown(f"**Metrics:** `artifacts/metrics/`")
    st.markdown(f"**Models:** `models/`")


# ============================================================
#  PAGE: PORTFOLIO OVERVIEW
# ============================================================

if page == "📊 Portfolio Overview":
    st.markdown("""
    <div class="main-header">
        <h1>🏦 Credit Risk Intelligence Engine</h1>
        <p>Real-time portfolio risk monitoring • Basel III compliant • ML-powered</p>
    </div>
    """, unsafe_allow_html=True)

    metrics = load_metrics(dataset)
    el = load_el_summary(dataset)

    if not metrics or not el:
        st.warning(f"⚠️ No metrics found for `{dataset}`. Run the training pipeline first.")
        st.code(f"python src/models/train_pd_model.py --dataset {dataset}")
        st.stop()

    # Find best model
    best_name = max(metrics, key=lambda k: metrics[k]["auc_roc"])
    best = metrics[best_name]

    # KPI Cards
    cols = st.columns(5)
    kpis = [
        ("Total Loans", f"{el['total_loans']:,}", "📋"),
        ("Portfolio EAD", f"${el['total_ead']:,.0f}", "💰"),
        ("Expected Loss", f"${el['total_expected_loss']:,.0f}", "⚡"),
        ("EL / EAD", f"{el['el_as_pct_of_ead']:.2f}%", "📊"),
        ("Best AUC-ROC", f"{best['auc_roc']:.4f}", "🎯"),
    ]

    for col, (label, value, icon) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 1.5rem;">{icon}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # Model comparison table
    st.markdown('<div class="section-header">📈 Model Comparison</div>', unsafe_allow_html=True)

    comparison_data = []
    for name, m in metrics.items():
        comparison_data.append({
            "Model": name,
            "AUC-ROC": m["auc_roc"],
            "Gini": m["gini"],
            "KS Statistic": m["ks_statistic"],
            "PR-AUC": m["pr_auc"],
            "F1 Score": m["f1_score"],
            "Precision": m["precision"],
            "Recall": m["recall"],
        })

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(
        comparison_df.style.highlight_max(
            subset=["AUC-ROC", "Gini", "KS Statistic", "PR-AUC", "F1 Score"],
            color="#6C63FF"
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Basel III EL Summary
    st.markdown('<div class="section-header">🏛️ Basel III Expected Loss</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average PD", f"{el['average_pd']:.4f}")
        st.metric("Median PD", f"{el['median_pd']:.4f}")
        st.metric("LGD Assumption", f"{el['lgd_assumption']:.0%}")

    with col2:
        st.metric("Fixed EAD", f"${el['fixed_ead_assumption']:,.0f}")
        st.metric("Total Expected Loss", f"${el['total_expected_loss']:,.0f}")
        st.metric("EL as % of EAD", f"{el['el_as_pct_of_ead']:.2f}%")


# ============================================================
#  PAGE: MODEL PERFORMANCE
# ============================================================

elif page == "🤖 Model Performance":
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Model Performance</h1>
        <p>ROC curves, Precision-Recall, KS statistic, and confusion matrix</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    roc_path = PLOTS_DIR / f"{dataset}_roc_curves.png"
    pr_path = PLOTS_DIR / f"{dataset}_pr_curves.png"
    ks_path = PLOTS_DIR / f"{dataset}_ks_curve.png"
    cm_path = PLOTS_DIR / f"{dataset}_confusion_matrix.png"
    pd_path = PLOTS_DIR / f"{dataset}_pd_distribution.png"

    with col1:
        if roc_path.exists():
            st.image(str(roc_path), caption="ROC Curve — All Models", use_container_width=True)
        else:
            st.info("ROC curve not generated yet.")

    with col2:
        if pr_path.exists():
            st.image(str(pr_path), caption="Precision-Recall Curve", use_container_width=True)
        else:
            st.info("PR curve not generated yet.")

    col3, col4 = st.columns(2)

    with col3:
        if ks_path.exists():
            st.image(str(ks_path), caption="KS Statistic", use_container_width=True)
        else:
            st.info("KS curve not generated yet.")

    with col4:
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion Matrix", use_container_width=True)
        else:
            st.info("Confusion matrix not generated yet.")

    if pd_path.exists():
        st.image(str(pd_path), caption="PD Score Distribution", use_container_width=True)


# ============================================================
#  PAGE: SHAP EXPLAINABILITY
# ============================================================

elif page == "🔍 SHAP Explainability":
    st.markdown("""
    <div class="main-header">
        <h1>🔍 SHAP Explainability</h1>
        <p>Regulatory-grade model explanations • Feature contributions</p>
    </div>
    """, unsafe_allow_html=True)

    shap_data = load_shap_importance(dataset)

    if shap_data:
        st.markdown('<div class="section-header">🏆 Top Features by SHAP Value</div>', unsafe_allow_html=True)

        # Top features table
        shap_df = pd.DataFrame([
            {"Feature": k, "Mean |SHAP|": v}
            for k, v in shap_data.items()
        ]).sort_values("Mean |SHAP|", ascending=False).head(15)

        st.dataframe(shap_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)

    beeswarm_path = PLOTS_DIR / f"{dataset}_shap_beeswarm.png"
    bar_path = PLOTS_DIR / f"{dataset}_shap_bar.png"
    waterfall_path = PLOTS_DIR / f"{dataset}_shap_waterfall.png"

    with col1:
        if beeswarm_path.exists():
            st.image(str(beeswarm_path), caption="SHAP Beeswarm (Global Feature Impact)", use_container_width=True)

    with col2:
        if bar_path.exists():
            st.image(str(bar_path), caption="Top 15 Features — Mean |SHAP|", use_container_width=True)

    if waterfall_path.exists():
        st.markdown('<div class="section-header">🔬 Single Loan Explanation (Waterfall)</div>', unsafe_allow_html=True)
        st.image(str(waterfall_path), caption="SHAP Waterfall — How one specific loan was scored", use_container_width=True)


# ============================================================
#  PAGE: ANOMALY DETECTION
# ============================================================

elif page == "⚠️ Anomaly Detection":
    st.markdown("""
    <div class="main-header">
        <h1>⚠️ Deep Learning Anomaly Detection</h1>
        <p>PyTorch Autoencoder + Isolation Forest ensemble</p>
    </div>
    """, unsafe_allow_html=True)

    anomaly_path = PLOTS_DIR / f"{dataset}_anomaly_reconstruction.png"

    if anomaly_path.exists():
        st.image(str(anomaly_path), caption="Autoencoder Reconstruction Error Distribution", use_container_width=True)

        st.markdown("""
        ### How to Read This Plot
        - **X-axis:** Log of reconstruction error. Loans on the **far right** are mathematically "weird"
        - **Green:** Healthy loans — cluster tightly on the left (low error)
        - **Orange:** Defaulted loans — spread further right on average
        - **Red dashed line:** Our anomaly cutoff. Anything past this line gets flagged

        ### Business Value
        Traditional credit scoring looks at individual features (income, DTI, credit score).
        The **Autoencoder** learns the deep mathematical *relationships* between features.
        When a loan's feature combinations don't make mathematical sense together
        (e.g., $500K income but zero credit lines), the reconstruction error spikes.

        This is how we catch **synthetic identity fraud** and **novel default patterns**
        that standard logistic regression would completely miss.
        """)
    else:
        st.warning(f"⚠️ No anomaly detection results for `{dataset}`. Run the DL pipeline first.")
        st.code(f"python src/models/train_dl.py --dataset {dataset} --model autoencoder")


# ============================================================
#  PAGE: RISK SEGMENTS
# ============================================================

elif page == "📋 Risk Segments":
    st.markdown("""
    <div class="main-header">
        <h1>📋 Basel III Risk Segmentation</h1>
        <p>Portfolio breakdown by PD-based risk grade (A through E)</p>
    </div>
    """, unsafe_allow_html=True)

    segments = load_el_segments(dataset)

    if not segments.empty:
        st.markdown('<div class="section-header">📊 Risk Grade Distribution</div>', unsafe_allow_html=True)
        st.dataframe(
            segments.style.format({
                "avg_pd": "{:.4f}",
                "total_ead": "${:,.0f}",
                "total_el": "${:,.0f}",
                "pct_of_portfolio": "{:.2f}%",
                "el_pct_of_ead": "{:.4f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )

    el_segment_path = PLOTS_DIR / f"{dataset}_el_by_segment.png"
    el_dist_path = PLOTS_DIR / f"{dataset}_el_distribution.png"

    col1, col2 = st.columns(2)
    with col1:
        if el_segment_path.exists():
            st.image(str(el_segment_path), caption="Risk Segmentation & Expected Loss by Grade", use_container_width=True)

    with col2:
        if el_dist_path.exists():
            st.image(str(el_dist_path), caption="Expected Loss Distribution per Loan", use_container_width=True)

    if not segments.empty:
        st.markdown('<div class="section-header">💡 Regulatory Insight</div>', unsafe_allow_html=True)
        st.markdown("""
        Under **Basel III / IV**, banks must hold capital reserves proportional to the
        Expected Loss (EL) of their loan portfolio:

        > **EL = PD × LGD × EAD**

        - **PD (Probability of Default):** Our ML model's prediction for each borrower
        - **LGD (Loss Given Default):** Assumed 45% for unsecured consumer credit
        - **EAD (Exposure at Default):** The outstanding loan balance at the time of default

        The risk grades (A through E) are assigned based on the model's predicted PD:
        | Grade | PD Range | Risk Level |
        |---|---|---|
        | **A** | 0% – 5% | Very Low |
        | **B** | 5% – 10% | Low |
        | **C** | 10% – 20% | Medium |
        | **D** | 20% – 50% | High |
        | **E** | 50% – 100% | Very High |
        """)
