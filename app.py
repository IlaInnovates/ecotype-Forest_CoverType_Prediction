import streamlit as st
import numpy as np
import pandas as pd
import pickle

# --- PAGE SETUP ---
st.set_page_config(page_title='EcoType: Forest Predictor', page_icon='🌲', layout='wide')

# --- LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    # Using the filenames from your train_model.py script
    model = pickle.load(open("best_forest_model.pkl", "rb"))
    le = pickle.load(open("label_encoder.pkl", "rb"))
    # This feature_names list has exactly 54 entries according to your error
    feature_names = pickle.load(open("model_features.pkl", "rb"))
    return model, le, feature_names

try:
    model, le, feature_names = load_artifacts()
    artifacts_ok = True
except Exception as e:
    st.error("⚠️ Model files not found. Run 'train_model.py' first.")
    artifacts_ok = False

# --- UI ---
st.title('🌲 EcoType: Forest Cover Type Prediction')

if artifacts_ok:
    st.subheader("Input Environmental Features")
    col1, col2 = st.columns(2)

    with col1:
        elevation = st.number_input("Elevation (meters)", value=2500.0)
        aspect = st.number_input("Aspect (0–360°)", value=100.0)
        slope = st.number_input("Slope (degrees)", value=10.0)
        h_hydro = st.number_input("Horiz. Dist to Hydrology (m)", value=100.0)
        v_hydro = st.number_input("Vert. Dist to Hydrology (m)", value=30.0)

    with col2:
        h_road = st.number_input("Horiz. Dist to Roadways (m)", value=1000.0)
        hill_9 = st.number_input("Hillshade 9am", value=210.0)
        hill_noon = st.number_input("Hillshade Noon", value=220.0)
        hill_3 = st.number_input("Hillshade 3pm", value=180.0)
        h_fire = st.number_input("Horiz. Dist to Fire Points (m)", value=1500.0)

    st.markdown("#### Area & Soil")
    wilderness = st.selectbox("Wilderness Area", ["Area 1", "Area 2", "Area 3", "Area 4"])
    soil_type = st.selectbox("Soil Type", [f"Soil Type {i}" for i in range(1, 41)])

    if st.button("Predict Cover Type"):
        # 1. CREATE THE BASE NUMERIC VECTOR (The first 10 columns)
        # Note: We do NOT add interaction terms here because the model expects 54 columns
        # (10 numeric + 4 wilderness + 40 soil = 54)
        num_feats = [elevation, aspect, slope, h_hydro, v_hydro, h_road, 
                     hill_9, hill_noon, hill_3, h_fire]

        # 2. CREATE ONE-HOT ENCODED BITS
        # 4 Wilderness Area bits
        wild_bits = [1.0 if f"Area {i+1}" == wilderness else 0.0 for i in range(4)]
        
        # 40 Soil Type bits
        soil_bits = [1.0 if f"Soil Type {i+1}" == soil_type else 0.0 for i in range(40)]

        # 3. COMBINE ALL (10 + 4 + 40 = 54)
        input_vector = num_feats + wild_bits + soil_bits
        
        # 4. PREDICTION
        # We wrap it in a DataFrame so column names match exactly
        input_df = pd.DataFrame([input_vector], columns=feature_names)
        
        # This will now be exactly 54 columns, matching your model!
        prediction_num = model.predict(input_df)[0]
        cover_name = le.inverse_transform([prediction_num])[0]

        st.success(f"✅ Predicted Forest Cover Type: **{cover_name}**")