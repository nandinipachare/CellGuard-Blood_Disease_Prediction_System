import os
import warnings
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

BASE_DIR = r"C:\Users\niles"
MODEL_PATH = os.path.join(BASE_DIR, "PR12_disease_prediction_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "PR12_prediction_results.csv")

st.set_page_config(
    page_title="CellGuard | Disease Prediction",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
<style>
.stApp {background: #f3f7fb;}
header[data-testid="stHeader"] {background: #f3f7fb;}
.block-container {padding-top: 2rem; padding-bottom: 3rem;}
.hero {
    background: linear-gradient(120deg, #102b46, #087e8b);
    padding: 30px; border-radius: 18px; color: white;
    margin-bottom: 22px;
}
.hero h1 {color: white; font-size: 36px; margin-bottom: 6px;}
.hero p {color: #e2f4f7; font-size: 16px;}
.panel {
    background: white; padding: 22px; border-radius: 15px;
    border: 1px solid #e0eaf0; margin-bottom: 16px;
}
.small-note {color: #5b7185; font-size: 13px;}
div.stButton > button[kind="primary"] {
    background: #087e8b; border: none; border-radius: 9px;
    padding: 0.65rem 1rem; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE_DIR, "PR12_prediction_results.csv"))

st.markdown("""
<div class="hero">
    <h1>🧬 CellGuard</h1>
    <p>Machine-learning-powered disease category prediction</p>
    <p>Analyze cell and laboratory features using your trained Random Forest pipeline.</p>
</div>
""", unsafe_allow_html=True)

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the saved model: {e}")
    st.stop()

# Load the original dataset to obtain valid input fields and categories.
# Keep the model's expected input columns in their original order.
dataset_path = os.path.join(BASE_DIR, "your_dataset.csv")

if not os.path.exists(dataset_path):
    st.warning(
        "Your original training dataset CSV is needed for the input form. "
        "Enter its filename below, or place a copy in C:\\Users\\niles."
    )
    dataset_filename = st.text_input(
        "Original dataset CSV filename",
        placeholder="e.g. blood_cell_dataset.csv"
    )
    if not dataset_filename:
        st.stop()
    dataset_path = os.path.join(BASE_DIR, dataset_filename)

try:
    df = pd.read_csv(dataset_path)
except Exception as e:
    st.error(f"Could not read the dataset: {e}")
    st.stop()

if hasattr(model, "feature_names_in_"):
    feature_cols = list(model.feature_names_in_)
else:
    excluded = {
        "disease_category", "anomaly_label", "cell_id",
        "cytodiffusion_anomaly_score",
        "cytodiffusion_classification_confidence",
        "labeller_confidence_score"
    }
    feature_cols = [c for c in df.columns if c not in excluded]

missing_cols = [c for c in feature_cols if c not in df.columns]
if missing_cols:
    st.error("The dataset is missing model input columns: " + ", ".join(missing_cols))
    st.stop()

st.markdown("## Prediction workspace")
st.write(
    "Enter the available cell and laboratory features below. "
    "The saved model pipeline performs its own preprocessing."
)

with st.form("prediction_form"):
    st.markdown("### Patient and cell information")
    st.caption("Values start with typical values from your dataset. Change them when you have suitable input data.")

    input_values = {}

    numeric_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(df[c])
    ]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    # Group inputs into sections for a cleaner form.
    groups = {
        "Cell morphology": [
            "cell_type", "cell_diameter_um", "nucleus_area_pct",
            "chromatin_density", "cytoplasm_ratio", "circularity",
            "eccentricity", "granularity_score", "lobularity_score",
            "membrane_smoothness", "cell_area_px", "perimeter_px"
        ],
        "Blood laboratory values": [
            "patient_age_group", "patient_sex", "wbc_count_per_ul",
            "rbc_count_millions_per_ul", "hemoglobin_g_dl",
            "hematocrit_pct", "platelet_count_per_ul", "mcv_fl",
            "mchc_g_dl"
        ],
        "Image and staining details": [
            "mean_r", "mean_g", "mean_b", "stain_intensity",
            "dataset_source", "staining_protocol", "microscope_model",
            "magnification_x", "image_resolution_px"
        ]
    }

    used = set()

    for section, preferred in groups.items():
        cols = [c for c in preferred if c in feature_cols]
        if not cols:
            continue

        with st.expander(section, expanded=(section == "Cell morphology")):
            left, right = st.columns(2)
            for i, col in enumerate(cols):
                container = left if i % 2 == 0 else right
                with container:
                    if col in numeric_cols:
                        values = pd.to_numeric(df[col], errors="coerce").dropna()
                        default = float(values.median()) if len(values) else 0.0
                        input_values[col] = st.number_input(
                            col.replace("_", " ").title(),
                            value=default,
                            key=f"input_{col}"
                        )
                    else:
                        values = df[col].dropna().astype(str).unique().tolist()
                        if not values:
                            values = ["Unknown"]
                        input_values[col] = st.selectbox(
                            col.replace("_", " ").title(),
                            values,
                            key=f"input_{col}"
                        )
                used.add(col)

    remaining = [c for c in feature_cols if c not in used]
    if remaining:
        with st.expander("Other model features", expanded=False):
            left, right = st.columns(2)
            for i, col in enumerate(remaining):
                container = left if i % 2 == 0 else right
                with container:
                    if col in numeric_cols:
                        values = pd.to_numeric(df[col], errors="coerce").dropna()
                        default = float(values.median()) if len(values) else 0.0
                        input_values[col] = st.number_input(
                            col.replace("_", " ").title(),
                            value=default,
                            key=f"input_{col}"
                        )
                    else:
                        values = df[col].dropna().astype(str).unique().tolist() or ["Unknown"]
                        input_values[col] = st.selectbox(
                            col.replace("_", " ").title(),
                            values,
                            key=f"input_{col}"
                        )

    submitted = st.form_submit_button(
        "Predict Disease Category",
        type="primary",
        use_container_width=True
    )

if submitted:
    try:
        input_df = pd.DataFrame(
            [[input_values[c] for c in feature_cols]],
            columns=feature_cols
        )
        prediction = model.predict(input_df)[0]

        st.markdown("---")
        st.markdown("## Prediction result")
        st.success(f"Predicted category: **{prediction}**")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = model.classes_
            results = pd.DataFrame({
                "Category": classes,
                "Model score (%)": (probabilities * 100).round(2)
            }).sort_values("Model score (%)", ascending=False)

            st.markdown("### Model scores by category")
            st.bar_chart(results.set_index("Category"))
            st.dataframe(results, use_container_width=True, hide_index=True)
            st.caption(
                "These scores reflect model output, not a clinically validated "
                "probability of having a disease."
            )
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.markdown("## Model performance")
st.caption("Metrics below are calculated from your saved test prediction results, if available.")

if os.path.exists(DATA_PATH):
    try:
        results_df = load_data()
        lower = {c.lower(): c for c in results_df.columns}
        actual_col = next(
            (lower[c] for c in lower if "actual" in c or "true" in c),
            None
        )
        pred_col = next(
            (lower[c] for c in lower if "predicted" in c or "prediction" in c),
            None
        )

        if actual_col and pred_col:
            actual = results_df[actual_col].astype(str)
            predicted = results_df[pred_col].astype(str)

            a, b, c, d = st.columns(4)
            a.metric("Accuracy", f"{accuracy_score(actual, predicted):.1%}")
            b.metric("Weighted precision", f"{precision_score(actual, predicted, average='weighted', zero_division=0):.1%}")
            c.metric("Weighted recall", f"{recall_score(actual, predicted, average='weighted', zero_division=0):.1%}")
            d.metric("Weighted F1", f"{f1_score(actual, predicted, average='weighted', zero_division=0):.1%}")
        else:
            st.info("Prediction CSV columns were not recognized for metric calculation.")
    except Exception as e:
        st.info(f"Could not calculate metrics from the results CSV: {e}")

st.markdown("""
<div class="panel">
<strong>Important medical disclaimer</strong><br>
This is an educational machine-learning project, not a medical diagnostic device.
Predictions may be inaccurate and must not be used to diagnose, treat, or rule out
any disease. Consult a qualified healthcare professional for medical decisions.
</div>
""", unsafe_allow_html=True)
