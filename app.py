# app.py 

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and feature names
model = joblib.load('heart_disease_model.pkl')
model_features = joblib.load('model_features.pkl')

st.title("❤️ Heart Disease Prediction App")

st.write("### Please enter the patient details:")

# Define user inputs
age = st.slider('Age', 18, 100, 50)
trestbps = st.slider('Resting Blood Pressure (mm Hg)', 80, 200, 120)
chol = st.slider('Cholesterol (mg/dl)', 100, 600, 200)
thalach = st.slider('Max Heart Rate', 60, 202, 150)
oldpeak = st.slider('Oldpeak (ST depression)', 0.0, 6.0, 1.0, 0.1)

sex = st.selectbox('Sex', ['Male', 'Female'])
cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['True', 'False'])
restecg = st.selectbox('Resting ECG', ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
exang = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])
slope = st.selectbox('ST Segment Slope', ['Upsloping', 'Flat', 'Downsloping'])
ca = st.selectbox('Number of Major Vessels (0-4)', [0, 1, 2, 3, 4])
thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])

# Base numerical features
user_input = {
    'age': age,
    'trestbps': trestbps,
    'chol': chol,
    'thalach': thalach,
    'oldpeak': oldpeak,
}

# All dummy columns set to 0 initially
for col in model_features:
    if col not in user_input:
        user_input[col] = 0

# Set the correct one-hot values
user_input[f"sex_{1 if sex == 'Male' else 0}"] = 1

cp_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
user_input[f"cp_{cp_map[cp]}"] = 1

user_input[f"fbs_{1 if fbs == 'True' else 0}"] = 1

restecg_map = {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2}
user_input[f"restecg_{restecg_map[restecg]}"] = 1

user_input[f"exang_{1 if exang == 'Yes' else 0}"] = 1

slope_map = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
user_input[f"slope_{slope_map[slope]}"] = 1

user_input[f"ca_{ca}"] = 1

thal_map = {'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3}
user_input[f"thal_{thal_map[thal]}"] = 1

# Create input DataFrame in exact order
input_df = pd.DataFrame([user_input])[model_features]

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("⚠️ High risk of heart disease.")
    else:
        st.success("✅ Low risk of heart disease.")
