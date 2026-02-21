"""
Streamlit web application for Lung Cancer Risk and Heart Failure Prediction.
Provides real-time predictions from both models.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.predict import predict, predict_batch, load_mlflow_model
from src.models.utils import get_feature_ranges

# Page configuration
st.set_page_config(
    page_title="Health Risk Prediction App",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-top: 1rem;
    }
    .risk-high {
        color: #ff4444;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .risk-medium {
        color: #ff8800;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .risk-low {
        color: #00aa00;
        font-weight: bold;
        font-size: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🏥 Health Risk Prediction Application</h1>', unsafe_allow_html=True)

# Sidebar for model selection
st.sidebar.title("Model Selection")
model_choice = st.sidebar.radio(
    "Select Prediction Model:",
    ["Lung Cancer Risk", "Heart Failure Prediction"]
)

# Lung Cancer Risk Prediction
if model_choice == "Lung Cancer Risk":
    st.header("🔬 Lung Cancer Risk Prediction")
    st.markdown("Enter patient information to predict lung cancer risk level.")
    
    # Tabs for single prediction and batch prediction
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
    
    with tab1:
        st.subheader("Patient Information")
        
        # Get feature ranges (for validation)
        try:
            data_path = Path("data/raw/lung_cancer.csv")
            if data_path.exists():
                feature_ranges = get_feature_ranges(str(data_path))
            else:
                feature_ranges = {}
        except Exception:
            feature_ranges = {}
        
        # Create input form in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=45, step=1)
            gender = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
            air_pollution = st.slider("Air Pollution", min_value=1, max_value=9, value=5, step=1)
            alcohol_use = st.slider("Alcohol Use", min_value=1, max_value=9, value=3, step=1)
            dust_allergy = st.slider("Dust Allergy", min_value=1, max_value=9, value=4, step=1)
            occupational_hazards = st.slider("Occupational Hazards", min_value=1, max_value=9, value=4, step=1)
            genetic_risk = st.slider("Genetic Risk", min_value=1, max_value=9, value=3, step=1)
        
        with col2:
            chronic_lung_disease = st.slider("Chronic Lung Disease", min_value=1, max_value=9, value=2, step=1)
            balanced_diet = st.slider("Balanced Diet", min_value=1, max_value=9, value=3, step=1)
            obesity = st.slider("Obesity", min_value=1, max_value=9, value=4, step=1)
            smoking = st.slider("Smoking", min_value=1, max_value=9, value=5, step=1)
            passive_smoker = st.slider("Passive Smoker", min_value=1, max_value=9, value=3, step=1)
            chest_pain = st.slider("Chest Pain", min_value=1, max_value=9, value=4, step=1)
            coughing_blood = st.slider("Coughing of Blood", min_value=1, max_value=9, value=2, step=1)
        
        with col3:
            fatigue = st.slider("Fatigue", min_value=1, max_value=9, value=3, step=1)
            weight_loss = st.slider("Weight Loss", min_value=1, max_value=9, value=2, step=1)
            shortness_breath = st.slider("Shortness of Breath", min_value=1, max_value=9, value=3, step=1)
            wheezing = st.slider("Wheezing", min_value=1, max_value=9, value=2, step=1)
            swallowing_difficulty = st.slider("Swallowing Difficulty", min_value=1, max_value=9, value=2, step=1)
            clubbing_nails = st.slider("Clubbing of Finger Nails", min_value=1, max_value=9, value=1, step=1)
            frequent_cold = st.slider("Frequent Cold", min_value=1, max_value=9, value=2, step=1)
            dry_cough = st.slider("Dry Cough", min_value=1, max_value=9, value=3, step=1)
            snoring = st.slider("Snoring", min_value=1, max_value=9, value=4, step=1)
        
        # Prediction button
        if st.button("🔮 Predict Risk Level", type="primary", use_container_width=True):
            try:
                # Prepare input data
                input_data = {
                    'Age': age,
                    'Gender': gender,
                    'Air Pollution': air_pollution,
                    'Alcohol use': alcohol_use,
                    'Dust Allergy': dust_allergy,
                    'OccuPational Hazards': occupational_hazards,
                    'Genetic Risk': genetic_risk,
                    'chronic Lung Disease': chronic_lung_disease,
                    'Balanced Diet': balanced_diet,
                    'Obesity': obesity,
                    'Smoking': smoking,
                    'Passive Smoker': passive_smoker,
                    'Chest Pain': chest_pain,
                    'Coughing of Blood': coughing_blood,
                    'Fatigue': fatigue,
                    'Weight Loss': weight_loss,
                    'Shortness of Breath': shortness_breath,
                    'Wheezing': wheezing,
                    'Swallowing Difficulty': swallowing_difficulty,
                    'Clubbing of Finger Nails': clubbing_nails,
                    'Frequent Cold': frequent_cold,
                    'Dry Cough': dry_cough,
                    'Snoring': snoring
                }
                
                # Make prediction
                with st.spinner("Predicting..."):
                    result = predict(input_data, return_proba=True)
                
                # Display result
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.subheader("Prediction Result")
                
                pred_label = result.get('prediction', 'Unknown')
                
                # Color code based on risk level
                if 'High' in pred_label:
                    st.markdown(f'<p class="risk-high">Risk Level: {pred_label}</p>', unsafe_allow_html=True)
                elif 'Medium' in pred_label:
                    st.markdown(f'<p class="risk-medium">Risk Level: {pred_label}</p>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<p class="risk-low">Risk Level: {pred_label}</p>', unsafe_allow_html=True)
                
                # Show probabilities if available
                if 'probabilities' in result and result['probabilities']:
                    st.subheader("Probability Scores")
                    st.markdown(
                        """
                        <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;'>
                        <p style='margin: 0; color: #666; font-size: 0.9rem;'>
                        <strong>What is this table?</strong><br>
                        This table shows the model's confidence scores for the prediction. 
                        The <strong>Confidence</strong> value indicates how certain the model is about its prediction 
                        (higher values = more confident). The <strong>Predicted Class</strong> is the risk level 
                        the model determined for this patient.
                        </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    prob_df = pd.DataFrame([result['probabilities']]).T
                    prob_df.columns = ['Value']
                    # Sort by value if numeric, otherwise keep order
                    try:
                        prob_df['NumericValue'] = prob_df['Value'].str.rstrip('%').astype(float)
                        prob_df = prob_df.sort_values('NumericValue', ascending=False)
                        prob_df = prob_df.drop('NumericValue', axis=1)
                    except:
                        prob_df = prob_df.sort_index()
                    st.dataframe(prob_df, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Please ensure the model has been trained. Run: `poetry run python src/models/train.py`")
    
    with tab2:
        st.subheader("Batch Prediction")
        st.markdown("Upload a CSV file with patient data to get predictions for multiple patients.")
        
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(df.head(), use_container_width=True)
                
                if st.button("🔮 Predict All", type="primary"):
                    with st.spinner("Processing predictions..."):
                        predictions = predict_batch(df)
                    
                    st.success("Predictions completed!")
                    st.subheader("Results")
                    st.dataframe(predictions, use_container_width=True)
                    
                    # Download button
                    csv = predictions.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name="lung_cancer_predictions.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

# Heart Failure Prediction (Placeholder)
else:
    st.header("❤️ Heart Failure Prediction")
    st.markdown("Enter patient information to predict heart failure risk.")
    st.info("⚠️ This is a placeholder. The actual model will be integrated when available from your teammate.")
    
    # Placeholder inputs (based on typical heart failure dataset features)
    col1, col2 = st.columns(2)
    
    with col1:
        age_hf = st.number_input("Age", min_value=0, max_value=120, value=60, step=1, key="hf_age")
        anaemia = st.selectbox("Anaemia", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="hf_anaemia")
        creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase", min_value=0, value=250, key="hf_cpk")
        diabetes = st.selectbox("Diabetes", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="hf_diabetes")
        ejection_fraction = st.slider("Ejection Fraction (%)", min_value=0, max_value=100, value=38, key="hf_ef")
        high_blood_pressure = st.selectbox("High Blood Pressure", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="hf_hbp")
    
    with col2:
        platelets = st.number_input("Platelets", min_value=0, value=263000, key="hf_platelets")
        serum_creatinine = st.number_input("Serum Creatinine", min_value=0.0, value=1.1, step=0.1, key="hf_creatinine")
        serum_sodium = st.number_input("Serum Sodium", min_value=0, max_value=200, value=136, key="hf_sodium")
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", key="hf_sex")
        smoking_hf = st.selectbox("Smoking", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="hf_smoking")
        time = st.number_input("Follow-up Period (days)", min_value=0, value=4, key="hf_time")
    
    if st.button("🔮 Predict Heart Failure Risk", type="primary", use_container_width=True):
        # Placeholder prediction
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.subheader("Prediction Result")
        st.markdown('<p class="risk-medium">Risk Level: No Heart Disease (Placeholder)</p>', unsafe_allow_html=True)
        st.info("This is a dummy prediction. The actual model will be integrated when available.")
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>MLOps Assignment - Health Risk Prediction System</p>
        <p>Built with Streamlit, PyCaret, and MLflow</p>
    </div>
    """,
    unsafe_allow_html=True
)
