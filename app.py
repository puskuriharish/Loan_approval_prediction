import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("loan_model.pkl")

st.title("🏦 Loan Approval Prediction App")

# Input fields matching the model
no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, step=1)
education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
self_employed = st.selectbox("Self Employed", ['Yes', 'No'])
income_annum = st.number_input("Annual Income (₹)", min_value=0)
loan_amount = st.number_input("Loan Amount (₹)", min_value=0)
loan_term = st.number_input("Loan Term (months)", min_value=1)
cibil_score = st.number_input("CIBIL Score (300-900)", min_value=300, max_value=900, value=700)
assets = st.number_input("Total Assets (₹)", min_value=0)

# Encode categorical fields
education = 1 if education == 'Graduate' else 0
self_employed = 1 if self_employed == 'Yes' else 0

# Combine features
features = np.array([[no_of_dependents, education, self_employed,
                      income_annum, loan_amount, loan_term,
                      cibil_score, assets]])

# Predict and display result
if st.button("Predict Loan Approval"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.success("✅ Loan Approved!")
    else:
        st.error("❌ Loan Rejected.")
