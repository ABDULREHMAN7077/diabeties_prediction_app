import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"

)
model = joblib.load("diabetes_model.pkl")
st.title("🩺 Diabetes Prediction App")
st.write(
    "This machine learning application estimates the risk of diabetes based on various health parameters.")
st.divider()
name = st.text_input("Enter The patients name")
st.subheader("👤 Patient Information")
gender = st.selectbox(
    "gender",
    options=["Male", "Female",]
)
age = st.number_input(
    "Age", min_value=0, max_value=120, value=25, step=10)
hypertension = st.selectbox(
    "Hypertension", ["Yes", "No"]   )
heart_disease = st.selectbox(
    "Heart Disease", ["Yes", "No"]  )
smoking_history = st.selectbox(
    "Smoking History", ["Never", "Formerly",])
bmi = st.number_input(
    "Body Mass Index (BMI)", min_value=0.0, max_value=100.0, value=25.0, step=1.0)
hba1c_level = st.number_input(
    "HbA1c Level", min_value=0.0, max_value=20.0, value=5.0, step=1.0  )
blood_glucose_level = st.number_input(
    "Blood Glucose Level", min_value=0.0, max_value=500.0, value=100.0, step=10.0)
 

st.divider()    
st.subheader("💉 Prediction")   
predict_button = st.button("Predict Diabetes Risk") 
if predict_button:
    input_data = np.array([[1 if gender == "Male" else 0, age,
     1 if hypertension == "Yes" else 0, 1 if heart_disease == "Yes" else 0,
    1 if smoking_history == "Formerly" else 0
     if smoking_history == "Never" else 2, bmi, hba1c_level, blood_glucose_level]])
    risk_percentage = model.predict_proba(input_data)[0][1] * 100
    st.spinner("Calculating risk percentage...")
    st.success(f"The predicted diabetes risk for {name} is: {risk_percentage:.2f}%")
    st.balloons()
    st.metric(label="Risk Percentage", value=f"{risk_percentage:.2f}%")
    st.toast("Prediction completed successfully!", icon="✅")
    
    st.info("Note: This prediction is based on a machine learning model and should not be considered a medical diagnosis. Please consult a healthcare professional for accurate assessment and advice. Otherwise we are not responsible for any health issues that may arise from the use of this application. ")
else:
    st.warning("Please enter the patient's information and click 'Predict Diabetes Risk' to get the prediction.")



st.divider()
with st.expander("ℹ️ About this App"):
    st.write(
        "This Diabetes Prediction App is a machine learning application that estimates the risk of diabetes based on various health parameters. "
    )