import streamlit as st
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer

st.set_page_config(
    page_title="CreditWise",
    page_icon="🏦",
    layout="centered"
)

def deep_patch_imputers(obj):
    """Recursively search and set _fill_dtype on all SimpleImputer instances."""
    if isinstance(obj, SimpleImputer) or obj.__class__.__name__ == "SimpleImputer":
        if not hasattr(obj, "_fill_dtype"):
            dtype = getattr(obj.statistics_, "dtype", None) if hasattr(obj, "statistics_") else None
            obj._fill_dtype = dtype if dtype is not None else "float64"
        return

    # Check Pipeline steps
    if hasattr(obj, "steps"):
        for _, step_obj in obj.steps:
            deep_patch_imputers(step_obj)
    if hasattr(obj, "named_steps"):
        for step_obj in obj.named_steps.values():
            deep_patch_imputers(step_obj)

    # Check ColumnTransformer transformers
    if hasattr(obj, "transformers_"):
        for item in obj.transformers_:
            if len(item) >= 2 and item[1] not in ("drop", "passthrough", None):
                deep_patch_imputers(item[1])
    if hasattr(obj, "transformers"):
        for item in obj.transformers:
            if len(item) >= 2 and item[1] not in ("drop", "passthrough", None):
                deep_patch_imputers(item[1])

@st.cache_resource
def load_model():
    m = joblib.load("model.joblib")
    deep_patch_imputers(m)
    return m

model = load_model()

st.title("🏦 CreditWise")
st.subheader("Loan Approval Prediction System")
st.write("Enter applicant details to predict the loan outcome.")

with st.form("loan_form"):
    st.header("Financial Details")

    applicant_income = st.number_input("Applicant Income", min_value=0.0, value=12000.0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=3000.0)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
    dti_ratio = st.number_input("DTI Ratio", min_value=0.0, max_value=1.0,
