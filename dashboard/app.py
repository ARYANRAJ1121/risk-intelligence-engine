"""
app.py — Credit Risk Intelligence Dashboard (v2 — Dynamic & Futuristic)
========================================================================
HOW TO RUN:
    streamlit run dashboard/app.py
"""

import sys
import json
import time
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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
#  THEME — FUTURISTIC DARK
# ============================================================
COLORS = {
    "bg_dark": "#0a0a1a",
    "bg_card": "#12122a",
    "bg_glass": "rgba(18, 18, 42, 0.7)",
    "accent_primary": "#6C63FF",
    "accent_secondary": "#00D4AA",
    "accent_warning": "#FF6B6B",
    "accent_gold": "#FFD93D",
    "text_primary": "#FFFFFF",
    "text_secondary": "rgba(255,255,255,0.6)",
    "gradient_1": "linear-gradient(135deg, #6C63FF, #3F3D8C)",
    "gradient_2": "linear-gradient(135deg, #00D4AA, #00876A)",
    "gradient_3": "linear-gradient(135deg, #FF6B6B, #C44545)",
    "gradient_hero": "linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%)",
}

PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "white", "family": "Inter"},
        "xaxis": {"gridcolor": "rgba(255,255,255,0.05)", "zerolinecolor": "rgba(255,255,255,0.05)"},
        "yaxis": {"gridcolor": "rgba(255,255,255,0.05)", "zerolinecolor": "rgba(255,255,255,0.05)"},
    }
}

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #0a0a1a; }

    /* Hero Header */
    .hero {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2.5rem 3rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(108,99,255,0.2);
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px; height: 400px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(108,99,255,0.15) 0%, transparent 70%);
    }
    .hero h1 {
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(90deg, #fff, #6C63FF);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0; letter-spacing: -1px;
    }
    .hero p { color: rgba(255,255,255,0.65); margin: 0.4rem 0 0; font-size: 1.05rem; }

    /* Glass Cards */
    .glass-card {
        background: rgba(18,18,42,0.65);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(108,99,255,0.15);
        border-radius: 16px;
        padding: 1.6rem;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(108,99,255,0.4);
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(108,99,255,0.15);
    }

    /* KPI Cards */
    .kpi-icon { font-size: 1.8rem; margin-bottom: 0.5rem; }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.9rem; font-weight: 700;
        background: linear-gradient(90deg, #6C63FF, #00D4AA);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .kpi-label {
        font-size: 0.75rem; color: rgba(255,255,255,0.45);
        text-transform: uppercase; letter-spacing: 1.5px; margin-top: 0.3rem;
    }
    .kpi-sub {
        font-size: 0.8rem; color: rgba(255,255,255,0.35);
        margin-top: 0.2rem; font-family: 'JetBrains Mono', monospace;
    }

    /* Section Headers */
    .section-title {
        font-size: 1.3rem; font-weight: 700; color: white;
        margin: 2.5rem 0 1rem; padding-bottom: 0.6rem;
        border-bottom: 2px solid rgba(108,99,255,0.4);
        display: inline-block;
    }

    /* Status Badges */
    .badge-ok {
        background: rgba(0,212,170,0.15); color: #00D4AA;
        padding: 4px 12px; border-radius: 20px; font-size: 0.75rem;
        font-weight: 600; display: inline-block;
    }
    .badge-warn {
        background: rgba(255,107,107,0.15); color: #FF6B6B;
        padding: 4px 12px; border-radius: 20px; font-size: 0.75rem;
        font-weight: 600; display: inline-block;
    }

    /* Sidebar */
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d20, #12122a);
        border-right: 1px solid rgba(108,99,255,0.15);
    }
    div[data-testid="stSidebar"] h2 { color: #6C63FF !important; }

    /* Table styling */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* Plotly chart containers */
    .plot-container { border-radius: 16px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
#  DATA LOADING
# ============================================================

@st.cache_data
def load_metrics(dataset):
    path = METRICS_DIR / f"{dataset}_model_metrics.json"
    return json.load(open(path)) if path.exists() else {}

@st.cache_data
def load_el_summary(dataset):
    path = METRICS_DIR / f"{dataset}_el_summary.json"
    return json.load(open(path)) if path.exists() else {}

@st.cache_data
def load_el_segments(dataset):
    path = METRICS_DIR / f"{dataset}_el_by_segment.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

@st.cache_data
def load_shap_importance(dataset):
    path = METRICS_DIR / f"{dataset}_shap_importance.json"
    return json.load(open(path)) if path.exists() else {}

@st.cache_data
def load_feature_data(dataset):
    from src.ingestion.config import Config
    path = Config.FEATURE_STORE_DIR / f"{dataset}_features.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


# ============================================================
#  PLOTLY CHART BUILDERS
# ============================================================

def make_gauge(value, title, max_val=1.0, color="#6C63FF", suffix=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 14, "color": "rgba(255,255,255,0.7)"}},
        number={"font": {"size": 32, "color": "white", "family": "JetBrains Mono"}, "suffix": suffix},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "rgba(255,255,255,0.3)", "tickfont": {"color": "rgba(255,255,255,0.4)"}},
            "bar": {"color": color, "thickness": 0.7},
            "bgcolor": "rgba(255,255,255,0.03)",
            "bordercolor": "rgba(255,255,255,0.1)",
            "steps": [
                {"range": [0, max_val * 0.5], "color": "rgba(0,212,170,0.07)"},
                {"range": [max_val * 0.5, max_val * 0.8], "color": "rgba(255,217,61,0.07)"},
                {"range": [max_val * 0.8, max_val], "color": "rgba(255,107,107,0.07)"},
            ],
            "threshold": {"line": {"color": "#FF6B6B", "width": 2}, "thickness": 0.8, "value": max_val * 0.85},
        },
    ))
    fig.update_layout(height=220, margin=dict(t=40, b=10, l=30, r=30), paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"})
    return fig


def make_radar(metrics, model_name):
    cats = ["AUC-ROC", "Gini", "KS", "Precision", "Recall", "F1"]
    vals = [metrics["auc_roc"], metrics["gini"], metrics["ks_statistic"],
            metrics["precision"], metrics["recall"], metrics["f1_score"]]
    vals.append(vals[0])  # close the polygon

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals, theta=cats + [cats[0]],
        fill="toself", fillcolor="rgba(108,99,255,0.2)",
        line=dict(color="#6C63FF", width=2),
        name=model_name,
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="rgba(255,255,255,0.4)")),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="rgba(255,255,255,0.7)")),
        ),
        showlegend=False, height=350,
        margin=dict(t=30, b=30, l=60, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_model_comparison_bar(metrics):
    models = list(metrics.keys())
    auc_vals = [metrics[m]["auc_roc"] for m in models]
    ks_vals = [metrics[m]["ks_statistic"] for m in models]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="AUC-ROC", x=models, y=auc_vals,
                         marker_color="#6C63FF", marker_line=dict(width=0)))
    fig.add_trace(go.Bar(name="KS Statistic", x=models, y=ks_vals,
                         marker_color="#00D4AA", marker_line=dict(width=0)))
    fig.update_layout(
        barmode="group", height=380,
        margin=dict(t=20, b=40, l=40, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white")),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Score"),
    )
    return fig


def make_shap_bar(shap_data, top_n=15):
    items = sorted(shap_data.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features = [x[0] for x in items][::-1]
    values = [x[1] for x in items][::-1]

    colors = [f"rgba(108,99,255,{0.4 + 0.6 * (i / len(features))})" for i in range(len(features))]

    fig = go.Figure(go.Bar(
        x=values, y=features, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.4f}" for v in values],
        textposition="outside", textfont=dict(color="rgba(255,255,255,0.7)", size=11),
    ))
    fig.update_layout(
        height=500, margin=dict(t=10, b=20, l=10, r=60),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Mean |SHAP Value|"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    return fig


def make_risk_donut(segments):
    grade_colors = {"A (Very Low)": "#00D4AA", "B (Low)": "#6C63FF", "C (Medium)": "#FFD93D", "D (High)": "#FF8C42", "E (Very High)": "#FF6B6B"}
    colors = [grade_colors.get(g, "#666") for g in segments["risk_grade"]]

    fig = go.Figure(go.Pie(
        labels=segments["risk_grade"], values=segments["loan_count"],
        hole=0.55, marker=dict(colors=colors, line=dict(color="#0a0a1a", width=3)),
        textinfo="label+percent", textfont=dict(size=12, color="white"),
        hovertemplate="<b>%{label}</b><br>Loans: %{value:,}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        height=380, margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
        showlegend=False,
        annotations=[dict(text="Risk<br>Grades", x=0.5, y=0.5, font_size=14, font_color="rgba(255,255,255,0.5)", showarrow=False)],
    )
    return fig


def make_el_waterfall(segments):
    fig = go.Figure(go.Waterfall(
        x=segments["risk_grade"], y=segments["total_el"],
        connector=dict(line=dict(color="rgba(255,255,255,0.1)")),
        increasing=dict(marker=dict(color="#FF6B6B")),
        decreasing=dict(marker=dict(color="#00D4AA")),
        totals=dict(marker=dict(color="#6C63FF")),
        text=[f"${v:,.0f}" for v in segments["total_el"]],
        textposition="outside", textfont=dict(color="rgba(255,255,255,0.7)", size=11),
    ))
    fig.update_layout(
        height=380, margin=dict(t=20, b=40, l=50, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Expected Loss ($)"),
    )
    return fig


def make_feature_explorer(df, target_col):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]
    if not numeric_cols or target_col not in df.columns:
        return None, []
    return df, numeric_cols


# ============================================================
#  SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## ⚡ Risk Engine")
    page = st.radio("Navigate", [
        "📊 Command Center",
        "🤖 Model Arena",
        "🔍 Explainability",
        "⚠️ Anomaly Radar",
        "📋 Risk Grid",
        "🔬 Data Explorer",
    ], label_visibility="collapsed")

    st.markdown("---")
    dataset = st.selectbox("Dataset", ["gmsc", "lc"],
        format_func=lambda x: "🏦 GMSC (150K)" if x == "gmsc" else "💳 LendingClub (1.3M)")

    st.markdown("---")
    st.markdown("##### System Status")
    model_exists = (MODELS_DIR / f"{dataset}_pd_model.pkl").exists()
    metrics_exist = (METRICS_DIR / f"{dataset}_model_metrics.json").exists()
    st.markdown(f'Model: <span class="{"badge-ok" if model_exists else "badge-warn"}">{"● Trained" if model_exists else "○ Missing"}</span>', unsafe_allow_html=True)
    st.markdown(f'Metrics: <span class="{"badge-ok" if metrics_exist else "badge-warn"}">{"● Ready" if metrics_exist else "○ Missing"}</span>', unsafe_allow_html=True)


# ============================================================
#  PAGE 1: COMMAND CENTER
# ============================================================

if page == "📊 Command Center":
    st.markdown("""
    <div class="hero">
        <h1>🏦 Credit Risk Intelligence</h1>
        <p>Real-time portfolio monitoring • Basel III compliant • ML + Deep Learning powered</p>
    </div>
    """, unsafe_allow_html=True)

    metrics = load_metrics(dataset)
    el = load_el_summary(dataset)

    if not metrics or not el:
        st.error(f"No data for `{dataset}`. Run: `python src/models/train_pd_model.py --dataset {dataset}`")
        st.stop()

    best_name = max(metrics, key=lambda k: metrics[k]["auc_roc"])
    best = metrics[best_name]

    # KPI Row
    cols = st.columns(5)
    kpis = [
        ("📋", "Total Loans", f"{el['total_loans']:,}", "test portfolio"),
        ("💰", "Portfolio EAD", f"${el['total_ead']:,.0f}", "exposure at default"),
        ("⚡", "Expected Loss", f"${el['total_expected_loss']:,.0f}", f"{el['el_as_pct_of_ead']:.2f}% of EAD"),
        ("🎯", "Best AUC-ROC", f"{best['auc_roc']:.4f}", best_name),
        ("🛡️", "KS Statistic", f"{best['ks_statistic']:.4f}", "discriminatory power"),
    ]
    for col, (icon, label, value, sub) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Gauges + Radar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.plotly_chart(make_gauge(best["auc_roc"], "AUC-ROC", 1.0, "#6C63FF"), use_container_width=True)
    with col2:
        st.plotly_chart(make_gauge(best["gini"], "Gini Coefficient", 1.0, "#00D4AA"), use_container_width=True)
    with col3:
        st.plotly_chart(make_gauge(best["ks_statistic"], "KS Statistic", 1.0, "#FFD93D"), use_container_width=True)
    with col4:
        st.plotly_chart(make_gauge(el["el_as_pct_of_ead"], "EL / EAD", 30, "#FF6B6B", "%"), use_container_width=True)

    # Model Comparison
    st.markdown('<div class="section-title">⚔️ Model Battle — Performance Comparison</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.plotly_chart(make_model_comparison_bar(metrics), use_container_width=True)
    with col2:
        st.plotly_chart(make_radar(best, best_name), use_container_width=True)

    # Detailed Table
    comparison_data = []
    for name, m in metrics.items():
        is_best = "🏆" if name == best_name else ""
        comparison_data.append({
            "": is_best, "Model": name,
            "AUC-ROC": m["auc_roc"], "Gini": m["gini"], "KS": m["ks_statistic"],
            "Precision": m["precision"], "Recall": m["recall"], "F1": m["f1_score"],
            "Train (s)": m.get("train_time_sec", 0),
        })
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)


# ============================================================
#  PAGE 2: MODEL ARENA
# ============================================================

elif page == "🤖 Model Arena":
    st.markdown("""
    <div class="hero">
        <h1>🤖 Model Performance Arena</h1>
        <p>Interactive evaluation charts • Click, zoom, and explore</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    roc_path = PLOTS_DIR / f"{dataset}_roc_curves.png"
    pr_path = PLOTS_DIR / f"{dataset}_pr_curves.png"
    ks_path = PLOTS_DIR / f"{dataset}_ks_curve.png"
    cm_path = PLOTS_DIR / f"{dataset}_confusion_matrix.png"
    pd_path = PLOTS_DIR / f"{dataset}_pd_distribution.png"

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if roc_path.exists():
            st.image(str(roc_path), caption="ROC Curve — All Models")
        else:
            st.info("ROC curve not generated yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if pr_path.exists():
            st.image(str(pr_path), caption="Precision-Recall Curve")
        else:
            st.info("PR curve not generated yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if ks_path.exists():
            st.image(str(ks_path), caption="KS Statistic")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion Matrix")
        st.markdown('</div>', unsafe_allow_html=True)

    if pd_path.exists():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.image(str(pd_path), caption="PD Score Distribution")
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
#  PAGE 3: EXPLAINABILITY
# ============================================================

elif page == "🔍 Explainability":
    st.markdown("""
    <div class="hero">
        <h1>🔍 SHAP Explainability Suite</h1>
        <p>Regulatory-grade model transparency • Why did the model decide that?</p>
    </div>
    """, unsafe_allow_html=True)

    shap_data = load_shap_importance(dataset)

    if shap_data:
        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown('<div class="section-title">🏆 Interactive Feature Importance</div>', unsafe_allow_html=True)
            st.plotly_chart(make_shap_bar(shap_data), use_container_width=True)

        with col2:
            st.markdown('<div class="section-title">📊 Top Features Table</div>', unsafe_allow_html=True)
            shap_df = pd.DataFrame([
                {"Rank": i+1, "Feature": k, "Mean |SHAP|": v}
                for i, (k, v) in enumerate(sorted(shap_data.items(), key=lambda x: x[1], reverse=True)[:15])
            ])
            st.dataframe(shap_df, use_container_width=True, hide_index=True)

    beeswarm_path = PLOTS_DIR / f"{dataset}_shap_beeswarm.png"
    waterfall_path = PLOTS_DIR / f"{dataset}_shap_waterfall.png"

    col1, col2 = st.columns(2)
    with col1:
        if beeswarm_path.exists():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.image(str(beeswarm_path), caption="SHAP Beeswarm — Global Feature Impact")
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if waterfall_path.exists():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.image(str(waterfall_path), caption="SHAP Waterfall — Single Loan Explanation")
            st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
#  PAGE 4: ANOMALY RADAR
# ============================================================

elif page == "⚠️ Anomaly Radar":
    st.markdown("""
    <div class="hero">
        <h1>⚠️ Deep Learning Anomaly Radar</h1>
        <p>PyTorch Autoencoder + Isolation Forest • Catching what standard models miss</p>
    </div>
    """, unsafe_allow_html=True)

    anomaly_path = PLOTS_DIR / f"{dataset}_anomaly_reconstruction.png"

    if anomaly_path.exists():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.image(str(anomaly_path), caption="Autoencoder Reconstruction Error Distribution")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="glass-card">
                <div class="kpi-icon">🧠</div>
                <div style="color:white; font-weight:600; font-size:1.1rem;">Autoencoder</div>
                <div class="kpi-sub">Learns the mathematical DNA of healthy loans. High reconstruction error = anomaly.</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="glass-card">
                <div class="kpi-icon">🌲</div>
                <div style="color:white; font-weight:600; font-size:1.1rem;">Isolation Forest</div>
                <div class="kpi-sub">Tree-based outlier detector. Isolates extreme single-feature anomalies.</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="glass-card">
                <div class="kpi-icon">🎯</div>
                <div style="color:white; font-weight:600; font-size:1.1rem;">Ensemble Flag</div>
                <div class="kpi-sub">When BOTH models agree → strongest fraud signal. Catches synthetic identities.</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.warning(f"Run: `python src/models/train_dl.py --dataset {dataset} --model autoencoder`")


# ============================================================
#  PAGE 5: RISK GRID
# ============================================================

elif page == "📋 Risk Grid":
    st.markdown("""
    <div class="hero">
        <h1>📋 Basel III Risk Segmentation Grid</h1>
        <p>Portfolio breakdown by PD-based risk grades (A through E)</p>
    </div>
    """, unsafe_allow_html=True)

    segments = load_el_segments(dataset)

    if not segments.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-title">🍩 Risk Grade Distribution</div>', unsafe_allow_html=True)
            st.plotly_chart(make_risk_donut(segments), use_container_width=True)

        with col2:
            st.markdown('<div class="section-title">📊 Expected Loss by Grade</div>', unsafe_allow_html=True)
            st.plotly_chart(make_el_waterfall(segments), use_container_width=True)

        st.markdown('<div class="section-title">📋 Detailed Segment Data</div>', unsafe_allow_html=True)
        st.dataframe(
            segments.style.format({
                "avg_pd": "{:.4f}", "total_ead": "${:,.0f}",
                "total_el": "${:,.0f}", "pct_of_portfolio": "{:.2f}%",
                "el_pct_of_ead": "{:.4f}%",
            }),
            use_container_width=True, hide_index=True,
        )

        # Basel III Explanation
        st.markdown("""
        <div class="glass-card" style="margin-top: 1rem;">
            <div style="color: white; font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;">
                🏛️ Basel III / IV Capital Requirement
            </div>
            <div style="color: rgba(255,255,255,0.6); line-height: 1.7;">
                <strong style="color: #6C63FF;">EL = PD × LGD × EAD</strong><br>
                <strong>PD</strong> = Probability of Default (our ML model prediction)<br>
                <strong>LGD</strong> = Loss Given Default (45% assumption for unsecured credit)<br>
                <strong>EAD</strong> = Exposure at Default (outstanding loan balance)<br><br>
                Banks must hold capital reserves ≥ Total Expected Loss to remain compliant.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No segment data available.")


# ============================================================
#  PAGE 6: DATA EXPLORER
# ============================================================

elif page == "🔬 Data Explorer":
    st.markdown("""
    <div class="hero">
        <h1>🔬 Live Data Explorer</h1>
        <p>Drill into the raw feature store • Interactive feature distributions</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_feature_data(dataset)

    if df.empty:
        st.warning("No feature data found.")
        st.stop()

    target_col = "SeriousDlqin2yrs" if dataset == "gmsc" else "is_default"

    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="glass-card" style="text-align:center;">
            <div class="kpi-value">{len(df):,}</div><div class="kpi-label">Total Rows</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="glass-card" style="text-align:center;">
            <div class="kpi-value">{len(df.columns)}</div><div class="kpi-label">Features</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        default_rate = df[target_col].mean() * 100 if target_col in df.columns else 0
        st.markdown(f"""<div class="glass-card" style="text-align:center;">
            <div class="kpi-value">{default_rate:.1f}%</div><div class="kpi-label">Default Rate</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        null_pct = df.isnull().mean().mean() * 100
        st.markdown(f"""<div class="glass-card" style="text-align:center;">
            <div class="kpi-value">{null_pct:.2f}%</div><div class="kpi-label">Missing Data</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Interactive feature distribution
    st.markdown('<div class="section-title">📊 Feature Distribution Explorer</div>', unsafe_allow_html=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]

    selected_feature = st.selectbox("Select a feature to explore:", numeric_cols)

    if selected_feature and target_col in df.columns:
        sample = df[[selected_feature, target_col]].dropna().sample(n=min(50000, len(df)), random_state=42)

        fig = go.Figure()
        for label, color, name in [(0, "#00D4AA", "Non-Default"), (1, "#FF6B6B", "Default")]:
            subset = sample[sample[target_col] == label][selected_feature]
            fig.add_trace(go.Histogram(
                x=subset, name=name, marker_color=color,
                opacity=0.6, nbinsx=60,
            ))

        fig.update_layout(
            barmode="overlay", height=400,
            title=f"Distribution of `{selected_feature}` by Default Status",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title=selected_feature),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Count"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Raw data preview
    st.markdown('<div class="section-title">📄 Raw Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)
