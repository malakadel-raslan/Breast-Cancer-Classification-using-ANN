import streamlit as st
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer ANN Classifier",
    page_icon="🔬",
    layout="wide",
)

# ── Load model & scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    import tensorflow as tf
    model = tf.keras.models.load_model("model.h5")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_names.pkl", "rb") as f:
        features = pickle.load(f)
    return model, scaler, features

model, scaler, FEATURES = load_assets()

# ── Feature metadata (min, mean, max) ────────────────────────────────────────
FEATURE_META = {
    "mean radius":              (6.98,  14.13, 28.11),
    "mean texture":             (9.71,  19.29, 39.28),
    "mean perimeter":           (43.79, 91.97, 188.5),
    "mean area":                (143.5, 654.9, 2501.0),
    "mean smoothness":          (0.053, 0.096, 0.163),
    "mean compactness":         (0.019, 0.104, 0.345),
    "mean concavity":           (0.0,   0.089, 0.427),
    "mean concave points":      (0.0,   0.049, 0.201),
    "mean symmetry":            (0.106, 0.181, 0.304),
    "mean fractal dimension":   (0.050, 0.063, 0.097),
    "radius error":             (0.112, 0.405, 2.873),
    "texture error":            (0.360, 1.217, 4.885),
    "perimeter error":          (0.757, 2.866, 21.98),
    "area error":               (6.80,  40.34, 542.2),
    "smoothness error":         (0.002, 0.007, 0.031),
    "compactness error":        (0.002, 0.026, 0.135),
    "concavity error":          (0.0,   0.032, 0.396),
    "concave points error":     (0.0,   0.012, 0.053),
    "symmetry error":           (0.008, 0.021, 0.079),
    "fractal dimension error":  (0.001, 0.004, 0.030),
    "worst radius":             (7.93,  16.27, 36.04),
    "worst texture":            (12.02, 25.68, 49.54),
    "worst perimeter":          (50.41, 107.3, 251.2),
    "worst area":               (185.2, 880.6, 4254.0),
    "worst smoothness":         (0.071, 0.132, 0.223),
    "worst compactness":        (0.027, 0.254, 1.058),
    "worst concavity":          (0.0,   0.272, 1.252),
    "worst concave points":     (0.0,   0.115, 0.291),
    "worst symmetry":           (0.157, 0.290, 0.664),
    "worst fractal dimension":  (0.055, 0.084, 0.208),
}

# ── Sidebar: sample cases ─────────────────────────────────────────────────────
SAMPLE_BENIGN = [
    13.54, 14.36, 87.46, 566.3, 0.0978, 0.0813, 0.0664, 0.0479, 0.1875, 0.0624,
    0.311, 1.197, 2.219, 24.56, 0.0056, 0.0219, 0.0209, 0.0131, 0.0212, 0.0031,
    15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.0736
]
SAMPLE_MALIGNANT = [
    20.57, 17.77, 132.9, 1326.0, 0.0847, 0.0786, 0.0869, 0.0702, 0.1812, 0.0567,
    0.541, 0.753, 3.399, 74.08, 0.0051, 0.0131, 0.0186, 0.0134, 0.013, 0.0022,
    24.99, 23.41, 158.8, 1956.0, 0.1238, 0.1866, 0.2416, 0.1860, 0.275, 0.0894
]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔬 Breast Cancer Classification")
st.markdown(
    "**Artificial Neural Network** trained on the UCI Breast Cancer Wisconsin Dataset  \n"
    "Enter the 30 cell-nucleus features below to receive a **Benign / Malignant** prediction."
)
st.divider()

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Quick Load")
    col1, col2 = st.columns(2)
    load_benign = col1.button("Load Benign Sample", use_container_width=True)
    load_malignant = col2.button("Load Malignant Sample", use_container_width=True)
    st.divider()
    st.markdown("### About the Model")
    st.markdown(
        "- **Architecture**: ANN (16 → 8 → 1)\n"
        "- **Regularisation**: Dropout (0.3)\n"
        "- **Optimizer**: Adam\n"
        "- **Loss**: Binary Cross-Entropy\n"
        "- **Test Accuracy**: ~98.3%\n"
        "- **Dataset**: 569 samples, 30 features"
    )
    st.divider()
    st.warning("⚠️ This tool is for **educational purposes only** and is not a medical device.")

# ── Initialise session state for values ───────────────────────────────────────
if "vals" not in st.session_state:
    st.session_state.vals = {f: FEATURE_META[f][1] for f in FEATURES}

if load_benign:
    for i, f in enumerate(FEATURES):
        st.session_state.vals[f] = SAMPLE_BENIGN[i]

if load_malignant:
    for i, f in enumerate(FEATURES):
        st.session_state.vals[f] = SAMPLE_MALIGNANT[i]

# ── Feature input in 3 groups ─────────────────────────────────────────────────
GROUPS = {
    "📏 Mean Features":  FEATURES[:10],
    "📐 Error Features": FEATURES[10:20],
    "⚠️ Worst Features": FEATURES[20:],
}

input_vals = {}

for group_title, group_features in GROUPS.items():
    st.subheader(group_title)
    cols = st.columns(5)
    for idx, feat in enumerate(group_features):
        lo, mean, hi = FEATURE_META[feat]
        step = round((hi - lo) / 200, 6)
        val = st.session_state.vals[feat]
        val = max(lo, min(hi, val))
        input_vals[feat] = cols[idx % 5].number_input(
            feat.replace("mean ", "").replace("worst ", "").replace(" error", " err").title(),
            min_value=float(lo),
            max_value=float(hi),
            value=float(round(val, 6)),
            step=float(step),
            format="%.4f",
            key=f"input_{feat}",
            help=f"Range: {lo} – {hi} | Mean: {mean}"
        )
    st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
predict_btn = st.button("🔍 Run Prediction", type="primary", use_container_width=True)

if predict_btn:
    x = np.array([[input_vals[f] for f in FEATURES]])
    x_scaled = scaler.transform(x)
    prob = float(model.predict(x_scaled, verbose=0)[0][0])
    label = "Malignant" if prob >= 0.5 else "Benign"

    st.divider()
    st.subheader("🧾 Prediction Result")

    res_col, prob_col = st.columns([1, 2])

    with res_col:
        if label == "Malignant":
            st.error(f"## 🔴 {label}", icon="🚨")
        else:
            st.success(f"## 🟢 {label}", icon="✅")
        st.metric("Malignancy Probability", f"{prob*100:.2f}%")
        st.metric("Benign Probability", f"{(1-prob)*100:.2f}%")

    with prob_col:
        import streamlit as st2
        st.markdown("#### Confidence Gauge")
        bar_html = f"""
        <div style="background:#e0e0e0;border-radius:8px;height:30px;width:100%;margin-bottom:8px">
          <div style="background:{'#e53935' if prob>=0.5 else '#43a047'};
                      width:{prob*100:.1f}%;height:100%;border-radius:8px;
                      transition:width 0.5s;display:flex;align-items:center;
                      padding-left:10px;color:white;font-weight:bold">
            {prob*100:.1f}%
          </div>
        </div>
        <p style="font-size:0.85em;color:#888">← Benign &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Malignant →</p>
        """
        st.markdown(bar_html, unsafe_allow_html=True)

        # Feature summary table
        st.markdown("#### Input Summary")
        summary = pd.DataFrame({
            "Feature": list(input_vals.keys()),
            "Value": [round(v, 4) for v in input_vals.values()]
        })
        st.dataframe(summary, use_container_width=True, height=200)
