import os
import pickle
import numpy as np
import streamlit as st

# Set up page config
st.set_page_config(
    page_title="Student Risk & Grade Predictor",
    page_icon="🎓",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title {
        color: #1E3A8A;
        font-weight: 800;
        text-align: center;
        margin-bottom: 8px;
    }
    .sub-title {
        color: #4B5563;
        text-align: center;
        margin-bottom: 24px;
    }
    .stButton>button {
        width: 100%;
        background-color: #2563EB;
        color: white;
        font-weight: bold;
        padding: 10px;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to load model safely
@st.cache_resource
def load_model():
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        st.error("Model file (`model.pkl`) not found in repository.")
        st.stop()
    with open(model_path, "rb") as file:
        return pickle.load(file)

model = load_model()

# Title Header
st.markdown("<h1 class='main-title'>🎓 Student Risk Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Adjust academic parameters below to estimate outcome using the trained classification model.</p>", unsafe_allow_html=True)

# Layout: Form Inputs & Summary Metrics
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("📚 Academic Marks Input")
    
    c1, c2 = st.columns(2)
    with c1:
        hindi = st.slider("Hindi Score", 0, 100, 75)
        english = st.slider("English Score", 0, 100, 80)
        science = st.slider("Science Score", 0, 100, 70)
    with c2:
        maths = st.slider("Maths Score", 0, 100, 85)
        history = st.slider("History Score", 0, 100, 65)
        geography = st.slider("Geography Score", 0, 100, 72)

    total_marks = hindi + english + science + maths + history + geography

with col2:
    st.subheader("📊 Score Summary")
    st.metric(label="Total Marks", value=f"{total_marks} / 600")
    st.metric(label="Overall Percentage", value=f"{(total_marks / 6):.2f}%")
    
    predict_btn = st.button("🚀 Predict Outcome")

# Inference logic on button click
if predict_btn:
    # Build input feature array to match model structure
    input_features = np.array([[hindi, english, science, maths, history, geography, total_marks]])
    
    try:
        prediction = model.predict(input_features)[0]
        
        st.divider()
        st.balloons()
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.success(f"**Predicted Result / Status:** {prediction}")
            
        with res_col2:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_features)[0]
                confidence = np.max(probs) * 100
                st.info(f"**Model Confidence:** {confidence:.2f}%")
            else:
                st.info("Inference completed successfully.")
                
    except Exception as e:
        st.error(f"Error making prediction: {e}")
