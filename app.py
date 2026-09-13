import os
import pickle
import numpy as np
import streamlit as st

# Set page layout and title
st.set_page_config(
    page_title="Student Risk Classifier",
    page_icon="🎓",
    layout="centered"
)

# Custom CSS styling embedded in Streamlit
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .stButton>button {
        width: 100%;
        background-color: #4f46e5;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #3730a3;
        color: white;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #e0e7ff;
        color: #3730a3;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Cache model loading to optimize performance
@st.cache_resource
def load_model():
    # Adjust file name if your pickled file is named logistic.pkl or model.pkl
    model_path = os.path.join(os.path.dirname(__file__), "logistic_2.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(__file__), "logistic.pkl")
    
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        return None

model = load_model()

st.title("🎓 Student Risk Assessment")
st.write("Fill in the student details below to predict their academic risk status.")

# Form interface
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=85.0)
        study_hours = st.number_input("Study Hours / Week", min_value=0.0, max_value=168.0, value=10.0)
        past_failures = st.number_input("Past Failures", min_value=0, max_value=20, value=0)
        assignments_completed_pct = st.number_input("Assignments Completed (%)", min_value=0.0, max_value=100.0, value=80.0)
        
        # Categorical Mappings
        parental_edu_label = st.selectbox(
            "Parental Education", 
            ["High School", "Bachelor", "Master", "Doctorate"]
        )
        parental_edu_map = {"High School": 0, "Bachelor": 1, "Master": 2, "Doctorate": 3}

    with col2:
        income_label = st.selectbox("Family Income", ["Low", "Medium", "High"])
        income_map = {"Low": 0, "Medium": 1, "High": 2}

        extracurricular_label = st.selectbox("Extracurricular Activities", ["No", "Yes"])
        extracurricular_map = {"No": 0, "Yes": 1}

        internet_label = st.selectbox("Internet Access", ["No", "Yes"])
        internet_map = {"No": 0, "Yes": 1}

        previous_grade = st.number_input("Previous Grade", min_value=0.0, max_value=100.0, value=75.0)
        final_score = st.number_input("Final Score", min_value=0.0, max_value=100.0, value=78.0)

    submit_button = st.form_submit_button("Predict Status")

# Handle Prediction
if submit_button:
    if model is None:
        st.error("Model could not be loaded. Please verify your `.pkl` file exists in the directory.")
    else:
        try:
            # Map categorical inputs to numbers
            features = [
                float(attendance),
                float(study_hours),
                float(past_failures),
                float(assignments_completed_pct),
                float(parental_edu_map[parental_edu_label]),
                float(income_map[income_label]),
                float(extracurricular_map[extracurricular_label]),
                float(internet_map[internet_label]),
                float(previous_grade),
                float(final_score),
            ]

            # Make prediction
            prediction = model.predict([features])[0]

            st.markdown(
                f'<div class="result-box">Predicted Category: {prediction}</div>', 
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
