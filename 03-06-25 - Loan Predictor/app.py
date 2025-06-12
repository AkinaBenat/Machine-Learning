import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("loan_model.pkl", "rb"))

st.title("🏦 Loan Approval Predictor")

# User input
Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
ApplicantIncome = st.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0.0)
LoanAmount = st.number_input("Loan Amount", min_value=0.0)
Loan_Amount_Term = st.number_input("Loan Amount Term", min_value=0.0)
Credit_History = st.selectbox("Credit History", [1.0, 0.0])
Property_Area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

# Preprocess input
input_data = pd.DataFrame({
    'Gender': [1 if Gender == 'Male' else 0],
    'Married': [1 if Married == 'Yes' else 0],
    'Education': [0 if Education == 'Graduate' else 1],
    'Self_Employed': [1 if Self_Employed == 'Yes' else 0],
    'ApplicantIncome': [ApplicantIncome],
    'CoapplicantIncome': [CoapplicantIncome],
    'LoanAmount': [LoanAmount],
    'Loan_Amount_Term': [Loan_Amount_Term],
    'Credit_History': [Credit_History],
    'Property_Area': [0 if Property_Area == 'Rural' else (1 if Property_Area == 'Semiurban' else 2)]
})

# Predict
if st.button("Predict Loan Status"):
    result = model.predict(input_data)[0]
    output = "✅ Approved" if result == 1 else "❌ Rejected"
    st.subheader(f"Result: {output}")
