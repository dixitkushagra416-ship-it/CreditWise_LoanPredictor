
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="CreditWise",
    page_icon="🏦",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

model = load_model()

st.title("🏦 CreditWise")
st.subheader("Loan Approval Prediction System")
st.write("Enter applicant details to predict the loan outcome.")

with st.form("loan_form"):
    st.header("Financial Details")

    applicant_income = st.number_input("Applicant Income", min_value=0.0, value=12000.0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=3000.0)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
    dti_ratio = st.number_input("DTI Ratio", min_value=0.0, max_value=1.0, value=0.3)
    existing_loans = st.number_input("Existing Loans", min_value=0, value=1)
    savings = st.number_input("Savings", min_value=0.0, value=10000.0)
    collateral_value = st.number_input("Collateral Value", min_value=0.0, value=20000.0)
    loan_amount = st.number_input("Loan Amount", min_value=0.0, value=15000.0)
    loan_term = st.number_input("Loan Term (months)", min_value=1, value=60)

    st.header("Personal Details")

    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    dependents = st.number_input("Dependents", min_value=0, value=0)

    employment_status = st.selectbox(
        "Employment Status",
        ["Salaried", "Self-employed", "Contract", "Unemployed"]
    )
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    education_level = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Home", "Car", "Education", "Business"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    employer_category = st.selectbox("Employer Category", ["Private", "Government", "MNC", "Business", "Unemployed"])

    submitted = st.form_submit_button("Predict Loan Approval")

if submitted:
    applicant = pd.DataFrame([{
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Employment_Status": employment_status,
        "Age": age,
        "Marital_Status": marital_status,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti_ratio,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Loan_Purpose": loan_purpose,
        "Property_Area": property_area,
        "Education_Level": education_level,
        "Gender": gender,
        "Employer_Category": employer_category
    }])

    prediction = model.predict(applicant)[0]

    st.subheader("Prediction Result")

    if str(prediction).lower() == "yes":
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Not Approved ❌")

    st.caption("Educational demonstration only. Not a real lending decision.")
