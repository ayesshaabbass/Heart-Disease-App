"""
Heart Disease Prediction System - Streamlit web app
Loads svc_trained_model.pkl (saved by the notebook, Step 6.4) and rebuilds the
same LabelEncoders from the same fixed category lists used in Step 5.1.
"""
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

MODEL_PATH = Path(__file__).parent / "svc_trained_model.pkl"

CATEGORIES = {
    "Gender": ["Male", "Female"],
    "ChestPain": ["Typical", "Atypical", "NonAnginal", "Asymptomatic"],
    "ExerciseAngina": ["Yes", "No"],
    "STSlope": ["Upsloping", "Flat", "Downsloping"],
}


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def build_label_encoders():
    # LabelEncoder codes categories alphabetically, so fitting on the same
    # lists reproduces the notebook's codes exactly.
    return {name: LabelEncoder().fit(values) for name, values in CATEGORIES.items()}


model = load_model()
encoders = build_label_encoders()

st.title("❤️ Heart Disease Prediction System")
st.markdown(
    "Built for **AIC354 Machine Learning Fundamentals (Assignment 2)**. "
    "Enter four clinical details and a Support Vector Classifier trained on the "
    "UCI Cleveland heart-disease data will predict whether heart disease is likely.\n\n"
    "*Accuracy on held-out test data: about **80%**. This is a learning demo, **not** a medical tool.*"
)
st.divider()

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", CATEGORIES["Gender"])
    chest = st.selectbox(
        "Chest Pain Type", CATEGORIES["ChestPain"], index=3,
        help="Typical / Atypical angina, Non-anginal pain, or Asymptomatic (no chest pain)",
    )
with col2:
    angina = st.selectbox("Exercise-Induced Angina", CATEGORIES["ExerciseAngina"], index=1)
    slope = st.selectbox("ST Slope (peak exercise)", CATEGORIES["STSlope"], index=1)

if st.button("Predict", type="primary", use_container_width=True):
    user_input = pd.DataFrame({
        "Gender": [gender], "ChestPain": [chest],
        "ExerciseAngina": [angina], "STSlope": [slope],
    })
    encoded = pd.DataFrame({c: encoders[c].transform(user_input[c]) for c in user_input.columns})
    prediction = model.predict(encoded)[0]

    st.divider()
    if prediction == 1:
        st.error("### ⚠️ HEART DISEASE LIKELY")
    else:
        st.success("### ✅ NO HEART DISEASE")

    with st.expander("See the encoded feature vector sent to the model"):
        st.dataframe(encoded, hide_index=True)

st.divider()
st.caption(
    "Trained on 302 patients (UCI Cleveland subset) using 4 input attributes only. "
    "The model misses roughly 1 in 3 real cases, so never use it for diagnosis."
)
