import os
import glob
import traceback

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="NetGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = "."
DATA_DIR = "data"
WINDOW_SIZE = 10

# ============================================================
# CUSTOM CSS - light professional cybersecurity theme
# ============================================================
CUSTOM_CSS = """
<style>
:root {
    --primary: #2563eb;
    --primary-light: #3b82f6;
    --success: #16a34a;
    --warning: #ea580c;
    --danger: #dc2626;
    --bg: #f4f6f9;
    --card-bg: #ffffff;
    --text-dark: #1f2937;
    --text-muted: #6b7280;
}

.stApp {
    background: var(--bg);
    color: var(--text-dark);
}

/* Force native Streamlit text containers (headers, widget labels, captions,
   selectboxes, expanders, sidebar nav) to a readable dark color - these
   inherit a light/white color from Streamlit's base theme by default,
   which disappears against this app's white cards/background.
   NOTE: these rules target the *container* elements only (not a blanket
   "div"/"span"/"p" selector), so inheritance fills in the text color
   without overriding this app's own custom-colored elements (risk-level
   text, metric values, tags), which set their color via a more specific
   class or an inline style further down the tree. */
[data-testid="stMarkdownContainer"],
[data-testid="stWidgetLabel"],
[data-testid="stCaptionContainer"],
[data-testid="stSelectbox"],
[data-testid="stNumberInput"],
[data-testid="stTextInput"],
[data-testid="stRadio"],
[data-testid="stExpander"],
.streamlit-expanderHeader,
[data-baseweb="select"],
section[data-testid="stSidebar"] {
    color: var(--text-dark) !important;
}

/* Streamlit's built-in top header/toolbar (the bar with Deploy / menu
   buttons) still uses Streamlit's dark default theme by default, which
   clashes with this app's light theme - recolor it to match. */
[data-testid="stHeader"] {
    background-color: #f4f6f9 !important;
}
[data-testid="stToolbar"] {
    background-color: transparent !important;
}
[data-testid="stHeader"] * {
    color: var(--text-dark) !important;
}
[data-testid="stHeader"] svg {
    fill: var(--text-dark) !important;
}

/* Sidebar background */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {
    background-color: #ffffff !important;
    border-right: 1px solid #e5e7eb;
}

/* ------------------------------------------------------------
   INPUT BOX FIX
   Streamlit's base theme renders number/text input boxes with a
   dark background by default. Previously only the typed-value text
   color was being forced dark, which made it invisible against that
   dark box (black-on-black). The rules below give the box itself a
   white background with a soft shadow, and keep the typed text dark
   so it's readable against the white box.
   ------------------------------------------------------------ */

/* The input box / wrapper itself -> white background + soft shadow */
[data-testid="stNumberInput"] > div > div,
[data-testid="stTextInput"] > div > div,
[data-baseweb="input"],
[data-baseweb="base-input"] {
    background-color: #ffffff !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.12) !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
}

/* The actual typed value inside number/text inputs -> stays dark, on transparent bg */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background-color: transparent !important;
    color: var(--text-dark) !important;
}

/* Keep dropdown popovers/menus (the list that opens on click) readable too */
[data-baseweb="popover"], [data-baseweb="menu"], ul[role="listbox"] {
    color: var(--text-dark) !important;
    background-color: #ffffff !important;
}

/* Hero section */
.hero {
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 55%, #3b82f6 100%);
    padding: 40px 32px;
    border-radius: 18px;
    color: black;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px rgba(37, 99, 235, 0.25);
}
.hero h1 {
    font-size: 2.3rem;
    margin-bottom: 4px;
    font-weight: 800;
}
.hero p {
    font-size: 1.05rem;
    opacity: 0.92;
    margin: 2px 0;
}
.badge-row {
    margin-top: 18px;
}
.badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.35);
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 0.85rem;
    margin-right: 10px;
    backdrop-filter: blur(4px);
    color: black;
}

/* Make Streamlit's column containers stretch to equal height, so cards
   placed inside sibling columns (e.g. the dashboard metric row and the
   model-card row) always match each other's height/width automatically. */
[data-testid="stHorizontalBlock"] {
    align-items: stretch;
}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    display: flex;
    flex-direction: column;
}
[data-testid="stColumn"] > div {
    height: 100%;
}
[data-testid="stColumn"] [data-testid="stVerticalBlock"] {
    height: 100%;
}

/* Generic cards */
.metric-card, .model-card, .result-card, .success-card, .threat-card, .info-card {
    background: var(--card-bg);
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
    border: 1px solid rgba(15, 23, 42, 0.05);
    box-sizing: border-box;
    width: 100%;
    height: 100%;
}

.metric-card {
    text-align: center;
    min-height: 110px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.metric-card .value {
    font-size: 1.9rem;
    font-weight: 800;
    color: var(--primary);
}
.metric-card .label {
    color: var(--text-muted);
    font-size: 0.9rem;
    margin-top: 4px;
}

.model-card {
    border-top: 4px solid var(--primary);
    margin-bottom: 16px;
    height: 195px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    overflow: hidden;
}
.model-card h4 {
    margin: 0 0 6px 0;
    color: var(--text-dark);
    font-size: 1.05rem;
    line-height: 1.25em;
    /* Reserve room for 2 lines of title text so single-line titles
       (e.g. "XGBoost") and two-line titles (e.g. "Random Forest")
       occupy the same vertical space, keeping every card the same height. */
    min-height: 2.5em;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.model-card p {
    color: var(--text-muted);
    font-size: 0.88rem;
    margin: 2px 0;
}
.model-card .tag {
    display: inline-block;
    margin-top: auto;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    align-self: flex-start;
}
.tag-blue { background: #dbeafe; color: #1e40af; }
.tag-green { background: #dcfce7; color: #166534; }
.tag-orange { background: #ffedd5; color: #9a3412; }

.result-card {
    text-align: center;
    padding: 28px;
    margin-top: 8px;
}
.success-card {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    border: 1px solid #86efac;
}
.threat-card {
    background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    border: 1px solid #fca5a5;
}
.info-card {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    border: 1px solid #93c5fd;
}
.success-card h2 { color: var(--success); margin-bottom: 4px; }
.threat-card h2 { color: var(--danger); margin-bottom: 4px; }

.status-dot-green { color: var(--success); font-size: 1.1rem; }
.status-dot-red { color: var(--danger); font-size: 1.1rem; }

.section-title {
    font-size: 1.35rem;
    font-weight: 800;
    color: var(--text-dark);
    margin: 18px 0 10px 0;
}

.footer {
    text-align: center;
    color: var(--text-muted);
    padding: 30px 0 10px 0;
    font-size: 0.85rem;
    border-top: 1px solid #e5e7eb;
    margin-top: 40px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================
# FEATURE DEFINITION (67 features - exact training order)
# ============================================================
FLOW_INFO_FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Flow Bytes/s",
    "Flow Packets/s",
]

FWD_BWD_FEATURES = [
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
]

TCP_FLAG_FEATURES = [
    "SYN Flag Count",
    "ACK Flag Count",
    "PSH Flag Count",
    "FIN Flag Count",
    "RST Flag Count",
    "URG Flag Count",
]

ADVANCED_FEATURES = [
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance",
    "CWE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean",
]

# Exact training feature order - the model ALWAYS expects this order.
FEATURE_COLUMNS = FLOW_INFO_FEATURES + FWD_BWD_FEATURES + TCP_FLAG_FEATURES + ADVANCED_FEATURES

assert len(FEATURE_COLUMNS) == 67, f"Expected 67 features, got {len(FEATURE_COLUMNS)}"

DEFAULT_VALUE = 0.0

MODEL_FILES = {
    "scaler_binary": "scaler_binary.pkl",
    "scaler_multi": "scaler_multi.pkl",
    "label_encoder": "label_encoder.pkl",
    "random_forest_binary": "random_forest_binary.pkl",
    "random_forest_multiclass": "random_forest_multiclass.pkl",
    "xgboost_binary": "xgboost_binary.pkl",
    "isolation_forest": "isolation_forest.pkl",
    "lstm_model": "lstm_model.keras",
    "autoencoder_model": "autoencoder_model.keras",
}

DISPLAY_NAMES = {
    "random_forest_binary": "Random Forest",
    "xgboost_binary": "XGBoost",
    "lstm_model": "LSTM",
    "isolation_forest": "Isolation Forest",
    "autoencoder_model": "Autoencoder",
    "random_forest_multiclass": "Random Forest Multiclass",
}

# ============================================================
# MODEL LOADING (cached, never trains)
# ============================================================
@st.cache_resource
def load_models():
    """Load all trained model artifacts from the models/ folder.
    Never trains or fits anything here. Missing files are reported,
    not treated as fatal errors."""
    artifacts = {}
    status = {}

    for key, filename in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(path):
            artifacts[key] = None
            status[key] = False
            continue
        try:
            if filename.endswith(".keras"):
                import tensorflow as tf
                artifacts[key] = tf.keras.models.load_model(path, compile=False)
            else:
                artifacts[key] = joblib.load(path)
            status[key] = True
        except Exception:
            artifacts[key] = None
            status[key] = False

    return artifacts, status


@st.cache_data
def find_dataset_path():
    """Look for a dataset csv inside data/. Returns None if not found."""
    if not os.path.isdir(DATA_DIR):
        return None
    csvs = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    return csvs[0] if csvs else None


@st.cache_data
def load_dataset(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data
def load_metrics():
    """Look for a metrics file (models/metrics.json). Never invents values."""
    path = os.path.join(MODEL_DIR, "metrics.json")
    if not os.path.exists(path):
        return None
    try:
        import json
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data
def find_autoencoder_threshold():
    """Optional threshold file for autoencoder decisioning. Not invented."""
    path = os.path.join(MODEL_DIR, "autoencoder_threshold.pkl")
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


artifacts, status = load_models()


def get_expected_columns(scaler, fallback):
    """Return the exact column names/order a fitted scaler expects.

    Scalers fit on a pandas DataFrame remember their input columns in
    `feature_names_in_`. That is the ONLY source of truth for what a
    given scaler needs - trusting a hardcoded feature list instead is
    what causes 'feature names should match those passed during fit'
    errors when the real training data used slightly different names
    (e.g. a duplicated 'Fwd Header Length.1' column, or 'ECE Flag Count'
    instead of 'CWE Flag Count'). If the scaler has no such attribute
    (e.g. it was fit on a bare numpy array), fall back to FEATURE_COLUMNS.
    """
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)
    return fallback


EXPECTED_BINARY_COLUMNS = get_expected_columns(artifacts.get("scaler_binary"), FEATURE_COLUMNS)
EXPECTED_MULTI_COLUMNS = get_expected_columns(artifacts.get("scaler_multi"), FEATURE_COLUMNS)

# ============================================================
# HELPERS
# ============================================================
def calculate_risk_level(probability):
    if probability < 0.30:
        return "LOW"
    elif probability < 0.60:
        return "MEDIUM"
    elif probability < 0.85:
        return "HIGH"
    else:
        return "CRITICAL"


RISK_COLORS = {
    "LOW": "#16a34a",
    "MEDIUM": "#ea580c",
    "HIGH": "#dc2626",
    "CRITICAL": "#991b1b",
    "Not available": "#6b7280",
}


def validate_features(df, expected_columns=FEATURE_COLUMNS):
    """Return list of columns the target model actually needs that are
    missing from df. Pass the specific scaler's EXPECTED_*_COLUMNS so
    this reflects what that model really requires, not just the
    hardcoded reference list."""
    return [c for c in expected_columns if c not in df.columns]


def align_features(df, expected_columns=FEATURE_COLUMNS):
    """Reindex df to exactly the column names/order a given scaler
    expects. Columns not needed are dropped; any needed-but-missing
    column is filled with 0.0 rather than raising a KeyError, so a
    naming mismatch between the UI's reference list and the scaler's
    real training columns degrades gracefully instead of crashing."""
    return df.reindex(columns=expected_columns, fill_value=0.0)


def status_icon(ok):
    return "🟢" if ok else "🔴"


def metric_card(value, label):
    st.markdown(
        f"""<div class="metric-card"><div class="value">{value}</div>
        <div class="label">{label}</div></div>""",
        unsafe_allow_html=True,
    )


def model_card(title, desc, tag, tag_class="tag-blue", icon="🧩"):
    st.markdown(
        f"""<div class="model-card">
        <h4>{icon} {title}</h4>
        <p>{desc}</p>
        <span class="tag {tag_class}">{tag}</span>
        </div>""",
        unsafe_allow_html=True,
    )


# ============================================================
# PREDICTION LOGIC
# ============================================================
def predict_binary(model_key, X_raw):
    """Runs a binary classifier (Random Forest / XGBoost). Returns dict."""
    model = artifacts.get(model_key)
    scaler = artifacts.get("scaler_binary")
    if model is None:
        return {"error": "Model unavailable. Please check the models folder."}
    if scaler is None:
        return {"error": "Binary scaler unavailable. Please check the models folder."}

    X = align_features(X_raw, EXPECTED_BINARY_COLUMNS)
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    prob = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]
        prob = float(np.max(proba))

    label = "ATTACK" if int(pred) == 1 else "BENIGN"
    return {
        "label": label,
        "probability": prob,
        "risk": calculate_risk_level(prob) if prob is not None else "Not available",
        "model": model,
        "X_scaled": X_scaled,
    }


def predict_lstm(X_raw):
    """LSTM needs WINDOW_SIZE consecutive rows. Only usable with a batch
    of at least WINDOW_SIZE rows."""
    model = artifacts.get("lstm_model")
    scaler = artifacts.get("scaler_binary")
    if model is None:
        return {"error": "Model unavailable. Please check the models folder."}
    if scaler is None:
        return {"error": "Binary scaler unavailable. Please check the models folder."}
    if len(X_raw) < WINDOW_SIZE:
        return {"error": f"⚠️ LSTM requires at least {WINDOW_SIZE} consecutive records."}

    X = align_features(X_raw, EXPECTED_BINARY_COLUMNS)
    X_scaled = scaler.transform(X)

    n_windows = len(X_scaled) - WINDOW_SIZE + 1
    sequences = np.array([X_scaled[i:i + WINDOW_SIZE] for i in range(n_windows)])

    preds = model.predict(sequences, verbose=0)
    preds = np.asarray(preds).reshape(len(preds), -1)
    probs = preds[:, -1] if preds.shape[1] == 1 else preds.max(axis=1)
    labels = ["ATTACK" if p >= 0.5 else "BENIGN" for p in probs]
    return {"labels": labels, "probabilities": probs}


def predict_unsupervised(model_key, X_raw):
    model = artifacts.get(model_key)
    scaler = artifacts.get("scaler_binary")
    if model is None:
        return {"error": "Model unavailable. Please check the models folder."}
    if scaler is None:
        return {"error": "Binary scaler unavailable. Please check the models folder."}

    X = align_features(X_raw, EXPECTED_BINARY_COLUMNS)
    X_scaled = scaler.transform(X)

    if model_key == "isolation_forest":
        raw_pred = model.predict(X_scaled)  # 1 normal, -1 anomaly
        scores = model.decision_function(X_scaled)
        labels = ["ATTACK" if p == -1 else "BENIGN" for p in raw_pred]
        return {"labels": labels, "scores": scores}

    if model_key == "autoencoder_model":
        reconstruction = model.predict(X_scaled, verbose=0)
        errors = np.mean(np.square(X_scaled - reconstruction), axis=1)
        threshold = find_autoencoder_threshold()
        if threshold is not None:
            labels = ["ATTACK" if e > threshold else "BENIGN" for e in errors]
        else:
            labels = ["Not available"] * len(errors)
        return {"labels": labels, "scores": errors, "threshold": threshold}

    return {"error": "Unsupported model."}


def predict_multiclass(X_raw):
    model = artifacts.get("random_forest_multiclass")
    scaler = artifacts.get("scaler_multi")
    encoder = artifacts.get("label_encoder")
    if model is None:
        return {"error": "Model unavailable. Please check the models folder."}
    if scaler is None:
        return {"error": "Multiclass scaler unavailable. Please check the models folder."}

    X = align_features(X_raw, EXPECTED_MULTI_COLUMNS)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)

    label = pred[0]
    if encoder is not None:
        try:
            label = encoder.inverse_transform([pred[0]])[0]
        except Exception:
            pass

    prob = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]
        prob = float(np.max(proba))

    return {
        "label": str(label),
        "probability": prob,
        "risk": calculate_risk_level(prob) if prob is not None else "Not available",
        "model": model,
        "X_scaled": X_scaled,
    }


def feature_importance_chart(model, top_n=10, expected_columns=None):
    if not hasattr(model, "feature_importances_"):
        return None
    importances = model.feature_importances_
    n = len(importances)

    # Prefer, in order: the model's own remembered training column names,
    # then the relevant scaler's real expected columns (EXPECTED_*_COLUMNS),
    # then the hardcoded reference list - but ONLY if the length actually
    # matches the importances array, since a length mismatch is what
    # crashes pd.DataFrame(). If nothing matches, fall back to generic
    # numbered labels, which are guaranteed to always match n and can
    # never raise "All arrays must be of the same length" again.
    candidates = []
    if hasattr(model, "feature_names_in_"):
        candidates.append(list(model.feature_names_in_))
    if expected_columns is not None:
        candidates.append(expected_columns)
    candidates.append(FEATURE_COLUMNS)

    feature_names = None
    for cand in candidates:
        if len(cand) == n:
            feature_names = cand
            break
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(n)]

    df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df = df.sort_values("Importance", ascending=False).head(top_n).iloc[::-1]
    fig = px.bar(
        df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Blues",
        title="Top Important Features",
    )
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        coloraxis_showscale=False, height=380,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.markdown("## 🛡️ NetGuard AI")
st.sidebar.caption("AI-Powered Network Traffic Anomaly Detection")
st.sidebar.divider()

PAGES = [
    "Dashboard",
    "Traffic Detection",
    "Batch Detection",
    "Model Comparison",
    "Dataset Analysis",
    "System Info",
]
page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")

st.sidebar.divider()
loaded_count = sum(1 for k in DISPLAY_NAMES if status.get(k))
st.sidebar.caption(f"Models loaded: {loaded_count}/6")


# ============================================================
# PAGE: DASHBOARD
# ============================================================
def page_dashboard():
    st.markdown(
        """<div class="hero">
        <h1>🛡️ NETGUARD AI</h1>
        <p><b>AI-Powered Network Traffic Anomaly Detection</b></p>
        <p>Detect suspicious network activity using Machine Learning and Deep Learning.</p>
        <div class="badge-row">
            <span class="badge">⚡ AI Detection</span>
            <span class="badge">🧠 Multi-Model</span>
            <span class="badge">🔐 Cybersecurity</span>
        </div>
        </div>""",
        unsafe_allow_html=True,
    )

    dataset_path = find_dataset_path()
    dataset = load_dataset(dataset_path) if dataset_path else None

    cols = st.columns(5 if dataset is not None else 4)
    with cols[0]:
        metric_card(loaded_count, "AI Models")
    with cols[1]:
        metric_card(len(FEATURE_COLUMNS), "Features")
    with cols[2]:
        metric_card(3, "Detection Types")
    with cols[3]:
        online = loaded_count > 0
        color = "status-dot-green" if online else "status-dot-red"
        st.markdown(
            f"""<div class="metric-card"><div class="value"><span class="{color}">●</span> {'Online' if online else 'Offline'}</div>
            <div class="label">System Status</div></div>""",
            unsafe_allow_html=True,
        )
    if dataset is not None:
        with cols[4]:
            n = len(dataset)
            display_val = f"{n/1000:.0f}K+" if n >= 1000 else str(n)
            metric_card(display_val, "Dataset Records")

    st.markdown('<div class="section-title">Available AI Models</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        model_card("Random Forest", "Binary classification", "BENIGN / ATTACK", "tag-blue", "🌲")
        model_card("Isolation Forest", "Unsupervised anomaly detection", "Anomaly Score", "tag-orange", "🌳")
    with c2:
        model_card("XGBoost", "Binary classification", "BENIGN / ATTACK", "tag-blue", "⚡")
        model_card("Autoencoder", "Deep learning anomaly detection", "Reconstruction Error", "tag-orange", "🧬")
    with c3:
        model_card("LSTM", "Deep learning binary classification", "BENIGN / ATTACK", "tag-blue", "🔁")
        model_card("Random Forest Multiclass", "Attack-type classification", "Multiclass", "tag-green", "🎯")


# ============================================================
# PAGE: TRAFFIC DETECTION
# ============================================================
def render_feature_inputs():
    """Renders the input form and returns a single-row DataFrame."""
    values = {}

    st.markdown("#### Flow Information")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        values["Destination Port"] = st.number_input("Destination Port", value=80, step=1)
    with c2:
        values["Flow Duration"] = st.number_input("Flow Duration", value=0.0)
    with c3:
        values["Flow Bytes/s"] = st.number_input("Flow Bytes/s", value=0.0)
    with c4:
        values["Flow Packets/s"] = st.number_input("Flow Packets/s", value=0.0)

    with st.expander("Forward / Backward Traffic", expanded=False):
        cols = st.columns(3)
        for i, feat in enumerate(FWD_BWD_FEATURES):
            with cols[i % 3]:
                values[feat] = st.number_input(feat, value=0.0, key=f"fb_{feat}")

    with st.expander("TCP Flags", expanded=False):
        cols = st.columns(3)
        for i, feat in enumerate(TCP_FLAG_FEATURES):
            with cols[i % 3]:
                values[feat] = st.number_input(feat, value=0, step=1, key=f"tcp_{feat}")

    with st.expander("⚙️ Advanced Features", expanded=False):
        cols = st.columns(3)
        for i, feat in enumerate(ADVANCED_FEATURES):
            with cols[i % 3]:
                values[feat] = st.number_input(feat, value=0.0, key=f"adv_{feat}")

    df = pd.DataFrame([values])
    return df[FEATURE_COLUMNS]


def page_traffic_detection():
    st.markdown("## 🔍 Traffic Detection")
    st.caption("Enter network traffic information and let the AI model determine "
               "whether the traffic is normal or suspicious.")

    detection_type = st.selectbox(
        "Detection Type",
        ["Binary Classification", "Unsupervised Detection", "Multiclass Classification"],
    )

    if detection_type == "Binary Classification":
        model_options = {"Random Forest": "random_forest_binary", "XGBoost": "xgboost_binary", "LSTM": "lstm_model"}
    elif detection_type == "Unsupervised Detection":
        model_options = {"Isolation Forest": "isolation_forest", "Autoencoder": "autoencoder_model"}
    else:
        model_options = {"Random Forest Multiclass": "random_forest_multiclass"}

    model_display = st.selectbox("Model", list(model_options.keys()))
    model_key = model_options[model_display]

    if not status.get(model_key):
        st.warning("⚠️ Model unavailable\n\nThe selected model could not be loaded. Please check the models folder.")

    if model_key == "lstm_model":
        st.info(f"⚠️ LSTM requires at least {WINDOW_SIZE} consecutive records. "
                "Use the **Batch Detection** page with a CSV of sequential traffic records.")

    st.divider()
    X_input = render_feature_inputs()

    analyze_disabled = (model_key == "lstm_model") or (not status.get(model_key))
    if st.button("🚀 Analyze Traffic", type="primary", use_container_width=True, disabled=analyze_disabled):
        try:
            if detection_type == "Binary Classification":
                result = predict_binary(model_key, X_input)
            elif detection_type == "Unsupervised Detection":
                result = predict_unsupervised(model_key, X_input)
                if "error" not in result:
                    result = {
                        "label": result["labels"][0],
                        "score": result["scores"][0],
                        "model": artifacts.get(model_key),
                        "X_scaled": None,
                        "model_key": model_key,
                    }
            else:
                result = predict_multiclass(X_input)

            if "error" in result:
                st.error(f"❌ {result['error']}")
                return

            render_prediction_result(detection_type, model_display, model_key, result, X_input)

        except Exception:
            st.error("❌ Something went wrong while analyzing this traffic record.")
            with st.expander("Technical details"):
                st.code(traceback.format_exc())


def render_prediction_result(detection_type, model_display, model_key, result, X_input):
    label = result.get("label", "Not available")

    if detection_type == "Unsupervised Detection":
        is_benign = label == "BENIGN"
    else:
        is_benign = label == "BENIGN"

    if label == "ATTACK":
        st.markdown(
            """<div class="result-card threat-card">
            <h2>🔴 THREAT DETECTED</h2>
            <p>The model classified this traffic as anomalous.</p>
            </div>""",
            unsafe_allow_html=True,
        )
    elif label == "BENIGN":
        st.markdown(
            """<div class="result-card success-card">
            <h2>🟢 TRAFFIC IS BENIGN</h2>
            <p>The model classified this traffic as normal.</p>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""<div class="result-card info-card">
            <h2>ℹ️ Result: {label}</h2>
            <p>Model prediction for this traffic record.</p>
            </div>""",
            unsafe_allow_html=True,
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card(model_display, "Model Used")
    with c2:
        if "probability" in result and result["probability"] is not None:
            metric_card(f"{result['probability']*100:.1f}%", "Confidence")
        elif "score" in result:
            metric_card(f"{result['score']:.4f}", "Anomaly Score")
        else:
            metric_card("Not available", "Probability")
    with c3:
        risk = result.get("risk", "Not available")
        color = RISK_COLORS.get(risk, "#777780")
        st.markdown(
            f"""<div class="metric-card"><div class="value" style="color:{color}">{risk}</div>
            <div class="label">Risk Level</div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">🧠 Model Explanation</div>', unsafe_allow_html=True)
    model_obj = result.get("model")

    if model_obj is not None and hasattr(model_obj, "feature_importances_"):
        fig = feature_importance_chart(model_obj)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
    elif model_key == "isolation_forest":
        c1, c2 = st.columns(2)
        with c1:
            metric_card(f"{result.get('score', 0):.4f}", "Anomaly Score")
        with c2:
            metric_card(label, "Decision")
    elif model_key == "autoencoder_model":
        metric_card(f"{result.get('score', 0):.6f}", "Reconstruction Error")
        st.caption("Threshold not available in models/ folder — showing the raw reconstruction "
                   "error only, no BENIGN/ATTACK threshold was applied.")
    else:
        st.info("Feature explanation is not available for this model.")


# ============================================================
# PAGE: BATCH DETECTION
# ============================================================
def page_batch_detection():
    st.markdown("## Batch Detection")
    st.caption("Upload a CSV file containing network-flow features and analyze multiple traffic records at once.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception:
        st.error("❌ Invalid CSV\n\nThe uploaded file could not be read.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card(len(df), "Rows")
    with c2:
        metric_card(df.shape[1], "Features")
    with c3:
        metric_card(int(df.isna().sum().sum()), "Missing Values")

    if df.isna().sum().sum() > 0:
        st.warning("⚠️ Missing Values\n\nThe uploaded dataset contains missing values. Please clean the data before prediction.")
        return

    detection_type = st.selectbox(
        "Detection Type",
        ["Binary Classification", "Unsupervised Detection", "Multiclass Classification"],
        key="batch_detection_type",
    )
    if detection_type == "Binary Classification":
        model_options = {"Random Forest": "random_forest_binary", "XGBoost": "xgboost_binary", "LSTM": "lstm_model"}
    elif detection_type == "Unsupervised Detection":
        model_options = {"Isolation Forest": "isolation_forest", "Autoencoder": "autoencoder_model"}
    else:
        model_options = {"Random Forest Multiclass": "random_forest_multiclass"}

    model_display = st.selectbox("Model", list(model_options.keys()), key="batch_model")
    model_key = model_options[model_display]

    if not status.get(model_key):
        st.warning("⚠️ Model unavailable\n\nThe selected model could not be loaded. Please check the models folder.")
        return

    # Validate against whichever scaler this specific model actually uses,
    # not a fixed reference list - binary/LSTM/unsupervised models use
    # scaler_binary, the multiclass model uses scaler_multi, and each may
    # expect slightly different column names.
    target_columns = EXPECTED_MULTI_COLUMNS if detection_type == "Multiclass Classification" else EXPECTED_BINARY_COLUMNS
    missing = validate_features(df, target_columns)
    if missing:
        st.error("❌ Missing Required Features")
        st.caption("The uploaded CSV is missing columns this model was trained on. "
                   "Missing values will default to 0, which may reduce accuracy:")
        st.write(missing)

    if st.button("🚀 Run Detection", type="primary", use_container_width=True):
        try:
            X = align_features(df, target_columns)

            if model_key == "lstm_model":
                res = predict_lstm(X)
                if "error" in res:
                    st.warning(res["error"])
                    return
                labels = res["labels"]
                confidences = list(res["probabilities"])
                risks = [calculate_risk_level(p) for p in confidences]
                result_df = pd.DataFrame({
                    "Record": range(1, len(labels) + 1),
                    "Prediction": labels,
                    "Confidence": [f"{c*100:.1f}%" for c in confidences],
                    "Risk Level": risks,
                })

            elif detection_type == "Binary Classification":
                scaler = artifacts.get("scaler_binary")
                model = artifacts.get(model_key)
                if scaler is None or model is None:
                    st.error("❌ Model unavailable\n\nThe selected model could not be loaded.")
                    return
                X_scaled = scaler.transform(X)
                preds = model.predict(X_scaled)
                labels = ["ATTACK" if int(p) == 1 else "BENIGN" for p in preds]
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_scaled)
                    confidences = proba.max(axis=1)
                    risks = [calculate_risk_level(c) for c in confidences]
                    conf_display = [f"{c*100:.1f}%" for c in confidences]
                else:
                    conf_display = ["Not available"] * len(labels)
                    risks = ["Not available"] * len(labels)
                result_df = pd.DataFrame({
                    "Record": range(1, len(labels) + 1),
                    "Prediction": labels,
                    "Confidence": conf_display,
                    "Risk Level": risks,
                })

            elif detection_type == "Unsupervised Detection":
                res = predict_unsupervised(model_key, X)
                if "error" in res:
                    st.error(f"❌ {res['error']}")
                    return
                labels = res["labels"]
                scores = res["scores"]
                score_label = "Anomaly Score" if model_key == "isolation_forest" else "Reconstruction Error"
                result_df = pd.DataFrame({
                    "Record": range(1, len(labels) + 1),
                    "Prediction": labels,
                    score_label: [f"{s:.4f}" for s in scores],
                })

            else:  # Multiclass
                scaler = artifacts.get("scaler_multi")
                model = artifacts.get(model_key)
                encoder = artifacts.get("label_encoder")
                if scaler is None or model is None:
                    st.error("❌ Model unavailable\n\nThe selected model could not be loaded.")
                    return
                X_scaled = scaler.transform(X)
                preds = model.predict(X_scaled)
                if encoder is not None:
                    try:
                        preds = encoder.inverse_transform(preds)
                    except Exception:
                        pass
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_scaled)
                    confidences = proba.max(axis=1)
                    conf_display = [f"{c*100:.1f}%" for c in confidences]
                else:
                    conf_display = ["Not available"] * len(preds)
                result_df = pd.DataFrame({
                    "Record": range(1, len(preds) + 1),
                    "Prediction": preds,
                    "Confidence": conf_display,
                })

            render_batch_results(result_df, detection_type)

        except Exception:
            st.error("❌ Something went wrong while running batch detection.")
            with st.expander("Technical details"):
                st.code(traceback.format_exc())


def render_batch_results(result_df, detection_type):
    total = len(result_df)
    benign = int((result_df["Prediction"] == "BENIGN").sum())
    attack = total - benign if "ATTACK" in result_df["Prediction"].values or benign <= total else 0
    attack_count = int((result_df["Prediction"] == "ATTACK").sum())
    anomaly_rate = (attack_count / total * 100) if total > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(total, "Total Records")
    with c2:
        metric_card(benign, "Benign")
    with c3:
        metric_card(attack_count, "Anomalies")
    with c4:
        metric_card(f"{anomaly_rate:.1f}%", "Anomaly Rate")

    if benign > 0 or attack_count > 0:
        st.markdown('<div class="section-title">Traffic Distribution</div>', unsafe_allow_html=True)
        dist_df = pd.DataFrame({
            "Class": ["BENIGN", "ATTACK"],
            "Count": [benign, attack_count],
        })
        fig = go.Figure(data=[go.Pie(
            labels=dist_df["Class"], values=dist_df["Count"], hole=0.55,
            marker=dict(colors=["#16a34a", "#dc2626"]),
        )])
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Results", data=csv, file_name="netguard_results.csv", mime="text/csv")


# ============================================================
# PAGE: MODEL COMPARISON
# ============================================================
def page_model_comparison():
    st.markdown("## 🤖 Model Comparison")

    metrics = load_metrics()
    model_names = ["Random Forest", "XGBoost", "LSTM", "Isolation Forest", "Autoencoder"]

    if not metrics:
        st.info("Evaluation metrics are not available.")
        st.caption("Place a `models/metrics.json` file with keys like "
                   "`{\"Random Forest\": {\"Accuracy\": 0.97, \"Precision\": ..., \"Recall\": ..., "
                   "\"F1-Score\": ..., \"ROC-AUC\": ...}}` to populate this page.")
        return

    rows = []
    for name in model_names:
        m = metrics.get(name)
        if m:
            rows.append({
                "Model": name,
                "Accuracy": m.get("Accuracy", "Not available"),
                "Precision": m.get("Precision", "Not available"),
                "Recall": m.get("Recall", "Not available"),
                "F1-Score": m.get("F1-Score", "Not available"),
                "ROC-AUC": m.get("ROC-AUC", "Not available"),
            })
    if not rows:
        st.info("Evaluation metrics are not available.")
        return

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    numeric_f1 = pd.to_numeric(df["F1-Score"], errors="coerce")
    if numeric_f1.notna().any():
        best_idx = numeric_f1.idxmax()
        best_model = df.loc[best_idx, "Model"]
        st.success(f"🏆 Best performing model by F1-Score: **{best_model}**")

        fig = px.bar(df, x="Model", y=numeric_f1, color="Model",
                     title="F1-Score Comparison", labels={"y": "F1-Score"})
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE: DATASET ANALYSIS
# ============================================================
def page_dataset_analysis():
    st.markdown("## 📊 Dataset Analysis")
    st.caption("Upload a CSV to analyze, or the app will use a dataset placed inside the data/ folder.")

    uploaded = st.file_uploader("Upload dataset CSV", type=["csv"], key="dataset_analysis_uploader")

    df = None
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception:
            st.error("❌ Invalid CSV\n\nThe uploaded file could not be read.")
            return
    else:
        dataset_path = find_dataset_path()
        if not dataset_path:
            st.info("Dataset not available.\n\nUpload a CSV above, or place the training dataset inside the data/ folder.")
            return
        df = load_dataset(dataset_path)
        if df is None:
            st.error("❌ Invalid CSV\n\nThe dataset file could not be read.")
            return

    label_col = None
    for candidate in ["Label", "label", "Class", "class"]:
        if candidate in df.columns:
            label_col = candidate
            break

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(len(df), "Total Records")
    with c2:
        metric_card(df.shape[1], "Number of Features")
    if label_col:
        benign_mask = df[label_col].astype(str).str.upper().str.contains("BENIGN")
        with c3:
            metric_card(int(benign_mask.sum()), "Benign Records")
        with c4:
            metric_card(int((~benign_mask).sum()), "Attack Records")
    else:
        with c3:
            metric_card("Not available", "Benign Records")
        with c4:
            metric_card("Not available", "Attack Records")

    if label_col:
        st.markdown('<div class="section-title">Class Distribution</div>', unsafe_allow_html=True)
        dist = df[label_col].value_counts().reset_index()
        dist.columns = ["Class", "Count"]
        fig = px.bar(dist, x="Class", y="Count", color="Class", title="Class Distribution")
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        st.markdown('<div class="section-title">Feature Distribution</div>', unsafe_allow_html=True)
        feature = st.selectbox("Select a feature", numeric_cols)
        fig = px.histogram(df, x=feature, nbins=40, title=f"Distribution of {feature}", color_discrete_sequence=["#2563eb"])
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=380)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Feature Statistics</div>', unsafe_allow_html=True)
        s = df[feature]
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            metric_card(f"{s.mean():.2f}", "Mean")
        with c2:
            metric_card(f"{s.median():.2f}", "Median")
        with c3:
            metric_card(f"{s.std():.2f}", "Std Dev")
        with c4:
            metric_card(f"{s.min():.2f}", "Minimum")
        with c5:
            metric_card(f"{s.max():.2f}", "Maximum")


# ============================================================
# PAGE: SYSTEM INFO
# ============================================================
def page_system_info():
    st.markdown("## ⚙️ System Information")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """<div class="info-card">
            <p><b>Framework:</b> Scikit-learn + XGBoost + TensorFlow</p>
            <p><b>Deployment:</b> Streamlit</p>
            <p><b>Input Features:</b> 67</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """<div class="info-card">
            <p><b>Models:</b> 6</p>
            <p><b>Detection:</b> Binary + Multiclass + Unsupervised</p>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Model Loading Status</div>', unsafe_allow_html=True)
    for key, display_name in DISPLAY_NAMES.items():
        ok = status.get(key, False)
        icon = "🟢" if ok else "🔴"
        state = "Loaded" if ok else "Missing"
        st.markdown(f"{icon} **{display_name}** — {state}")

    st.markdown('<div class="section-title">Feature Alignment</div>', unsafe_allow_html=True)
    st.caption("Each scaler's real expected feature names (from `feature_names_in_`), compared against "
               "the app's hardcoded reference list. Differences here are auto-corrected at prediction "
               "time (missing columns default to 0), but any listed here means the UI form doesn't collect "
               "that field yet.")
    for scaler_key, expected in [("scaler_binary", EXPECTED_BINARY_COLUMNS), ("scaler_multi", EXPECTED_MULTI_COLUMNS)]:
        scaler_obj = artifacts.get(scaler_key)
        with st.expander(f"{scaler_key} — {len(expected)} expected features"):
            if scaler_obj is None:
                st.write("Scaler not loaded.")
                continue
            extra_in_reference = sorted(set(FEATURE_COLUMNS) - set(expected))
            missing_from_reference = sorted(set(expected) - set(FEATURE_COLUMNS))
            if not extra_in_reference and not missing_from_reference:
                st.success("Reference list matches the scaler's expected features exactly.")
            else:
                if missing_from_reference:
                    st.warning("Expected by scaler but not collected by the input form (defaults to 0):")
                    st.write(missing_from_reference)
                if extra_in_reference:
                    st.info("Collected by the input form but ignored by this scaler:")
                    st.write(extra_in_reference)


# ============================================================
# ROUTER
# ============================================================
if page == "Dashboard":
    page_dashboard()
elif page == "Traffic Detection":
    page_traffic_detection()
elif page == "Batch Detection":
    page_batch_detection()
elif page == "Model Comparison":
    page_model_comparison()
elif page == "Dataset Analysis":
    page_dataset_analysis()
elif page == "System Info":
    page_system_info()

st.markdown(
    """<div class="footer">
    ────────────────────────────<br>
    🛡️ <b>NetGuard AI</b><br>
    AI-Powered Network Traffic Anomaly Detection<br>
    Built with Python • Scikit-learn • XGBoost • TensorFlow • Streamlit<br>
    ────────────────────────────
    </div>""",
    unsafe_allow_html=True,
)