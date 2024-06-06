# -*- coding: utf-8 -*-
"""
Created on Wed May 29 05:55:37 2024
@author: Navya
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Get the current working directory
cwd = os.getcwd()

# Specify the paths to the files relative to the current working directory
model_file_path = os.path.join(cwd, "E:\loan\loan_approval_model.pkl")
scaler_file_path = os.path.join(cwd, "E:\loan\scaler.pkl")
label_encoder_file_path = os.path.join(cwd, "E:\loan\label_encoder.pkl")

# Load the model
with open(model_file_path, 'rb') as f:
    model = pickle.load(f)

# Load the scaler
with open(scaler_file_path, 'rb') as f:
    scaler = pickle.load(f)

# Load the label encoder
with open(label_encoder_file_path, 'rb') as f:
    le = pickle.load(f)

# Function to get user input and predict loan approval
def predict_loan_approval(ApplicantIncome, CoapplicantIncome, LoanAmount, Credit_History, Property_Area):
    user_data = {
        'ApplicantIncome': ApplicantIncome,
        'CoapplicantIncome': CoapplicantIncome,
        'LoanAmount': LoanAmount,
        'Credit_History': Credit_History,
        'Property_Area': Property_Area
    }
    
    # Convert input to DataFrame
    user_df = pd.DataFrame(user_data, index=[0])
    
    # Feature engineering
    user_df['TotalIncome'] = user_df['ApplicantIncome'] + user_df['CoapplicantIncome']
    user_df['IncomeLoanRatio'] = user_df['TotalIncome'] / user_df['LoanAmount']
    
    # Ensure the DataFrame has the same structure as the training data
    user_df = user_df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Credit_History', 'Property_Area', 'TotalIncome', 'IncomeLoanRatio']]
    
    # Scale the user data
    user_data_scaled = scaler.transform(user_df)
    
    # Predict loan approval
    prediction = model.predict(user_data_scaled)
    
    return 'Approved' if prediction[0] == 1 else 'Not Approved'

# Streamlit interface
st.title("Loan Approval Prediction App")

ApplicantIncome = st.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
LoanAmount = st.number_input("Loan Amount", min_value=0)
Credit_History = st.selectbox("Credit History", [0, 1])
Property_Area = st.selectbox("Property Area", le.classes_)

if st.button("Predict"):
    Property_Area_encoded = list(le.transform([Property_Area]))[0]
    result = predict_loan_approval(ApplicantIncome, CoapplicantIncome, LoanAmount, Credit_History, Property_Area_encoded)
    st.write(f"Loan Status: {result}")
