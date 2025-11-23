import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# ---------- App Title and Instructions ----------
st.set_page_config(page_title="EcoType Forest Cover", page_icon="🌲", layout="centered")
st.title("🔟 Final Task: EcoType Forest Cover Streamlit UI")
st.markdown("""
**Design a clean, interactive Streamlit application!**

- Manually input values for each feature below.
- Click "Predict" to see the forest cover type.
- Output will be human-friendly and inverse-mapped for clarity.
""")

# ---------- Load model artifacts ----------
models_dir = Path("models")
default_model_path = models_dir / "best_model.pkl"
default_enc_path = models_dir / "target_encoder.pkl"

with st.sidebar:
    st.header("Artifact & Feature Info")
    model_path = Path(st.text_input("Model path", str(default_model_path)))
    enc_path = Path(st.text_input("Encoder path", str(default_enc_path)))

missing = []
if not model_path.exists():
    missing.append(f"Missing: {model_path}")
if not enc_path.exists():
    missing.append(f"Missing: {enc_path}")
if missing:
    st.error("Missing artifacts:\n" + "\n".join(missing))
    st.stop()

try:
    with open(model_path, "rb") as f:
        model_bundle = pickle.load(f)
    with open(enc_path, "rb") as f:
        encoder_bundle = pickle.load(f)
except Exception as e:
    st.error(f"Artifact Read Error: {e}")
    st.stop()

model = model_bundle["model"]
FEATURES = list(model_bundle["features"])
NUMERIC_COLS = [c for c in model_bundle.get("numeric_cols", []) if c in FEATURES]
WILDERNESS_COLS = [c for c in model_bundle.get("wilderness_cols", []) if c in FEATURES]
SOIL_COLS = [c for c in model_bundle.get("soil_cols", []) if c in FEATURES]
ENGINEERED_COLS = [c for c in FEATURES if c not in set(NUMERIC_COLS + WILDERNESS_COLS + SOIL_COLS)]
id_to_name = dict(encoder_bundle.get("id_to_name", {}))

with st.sidebar:
    st.subheader("Expected Features (X_train columns)")
    st.write(f"Total features: {len(FEATURES)}")
    st.text_area("Features", value="\n".join(FEATURES), height=260)
    show_preview = st.checkbox("Show input preview?", value=True)

# ---------- Feature Input UI ----------
st.header("Step 1: Enter Feature Values")
st.subheader("Numeric Features")
num_vals = {}
for col in NUMERIC_COLS:
    num_vals[col] = st.number_input(col, value=0.0, step=1.0, format="%.2f")

st.subheader("Wilderness Area (One-Hot Flags)")
wild_vals = {col: st.selectbox(col, options=[0, 1], index=0) for col in WILDERNESS_COLS}

st.subheader("Soil Type (One-Hot Flags)")
soil_vals = {col: st.selectbox(col, options=[0, 1], index=0) for col in SOIL_COLS}

if ENGINEERED_COLS:
    st.subheader("Engineered Features")
    eng_vals = {col: st.number_input(col, value=0.0, step=1.0, format="%.2f") for col in ENGINEERED_COLS}
else:
    eng_vals = {}

# ---------- Predict and Display Output ----------
st.header("Step 2: Predict Forest Cover Type")

col_left, col_right = st.columns([1, 1])
with col_left:
    predict_clicked = st.button("Predict", type="primary")
with col_right:
    if st.button("Reset"): st.experimental_rerun()

if predict_clicked:
    inputs = {**num_vals, **wild_vals, **soil_vals, **eng_vals}
    missing_cols = [c for c in FEATURES if c not in inputs]
    extra_cols = [c for c in inputs if c not in FEATURES]

    if missing_cols:
        st.error("Missing features: " + ", ".join(missing_cols))
        st.stop()
    if extra_cols:
        st.warning("Ignoring extra inputs: " + ", ".join(extra_cols))

    df_in = pd.DataFrame([inputs], columns=FEATURES)

    if show_preview:
        st.info("Preview of input row (matches X_train):")
        st.dataframe(df_in, use_container_width=True)

    try:
        pred_id = int(model.predict(df_in)[0])
        pred_name = id_to_name.get(pred_id, f"Class {pred_id}")
        st.success(f"🌲 Predicted Cover Type: **{pred_name}**")
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.info("Check model pipeline/preprocessing and column alignment.")

# ---------- Probabilities (Optional) ----------
with st.expander("Show Class Probabilities"):
    try:
        df_cur = pd.DataFrame([{**num_vals, **wild_vals, **soil_vals, **eng_vals}], columns=FEATURES)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df_cur)[0]
            class_ids = getattr(model, "classes_", list(range(len(probs))))
            class_names = [id_to_name.get(int(cid), str(cid)) for cid in class_ids]
            st.bar_chart(pd.DataFrame({"Probability": probs}, index=class_names))
        else:
            st.write("Model does not have predict_proba.")
    except Exception:
        st.write("Input values required—click Predict above first.")

# --- End of App ---
