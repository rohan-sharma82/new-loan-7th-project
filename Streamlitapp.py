import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="Bank Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to inject a clean financial styling look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #1e3d59;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 10px 0px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #17b978;
        color: white;
    }
    h1, h2, h3 {
        color: #1e3d59;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to cache pipeline loading
@st.cache_resource
def load_pipeline(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Pipeline file '{filepath}' not found.")
    with open(filepath, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline

# App header
st.title("🏦 Underwriter Decision Support Dashboard")
st.markdown("""
*Evaluate loan approval probability instantly based on predictive historical credit profiles.*

**Instructions:** Provide the client demographics and financial details below, then click the evaluation button at the bottom of the page to run the automated credit diagnosis.
""")
st.write("---")

# Try to load the pipeline
pipeline_file = 'loan_approval_pipeline.pkl'
pipeline_loaded = False

try:
    pipeline = load_pipeline(pipeline_file)
    pipeline_loaded = True
except Exception as e:
    st.error("⚠️ System Configuration Error: Missing Model Artifact")
    st.info(f"Please ensure **'{pipeline_file}'** is located in the working directory before deploying this app. Error details: {str(e)}")

# Form Layout
with st.form("loan_application_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Demographics & Background")
        
        gender = st.selectbox("Gender", options=["Male", "Female"], help="Select the applicant's biological gender.")
        married = st.selectbox("Married", options=["Yes", "No"], help="Marital status of the main applicant.")
        dependents = st.selectbox("Dependents", options=["0", "1", "2", "3+"], help="Number of dependent family members.")
        education = st.selectbox("Education", options=["Graduate", "Not Graduate"], help="Education status of primary applicant.")
        self_employed = st.selectbox("Self Employed", options=["No", "Yes"], help="Is the applicant self-employed?")
        property_area = st.selectbox("Property Area", options=["Semiurban", "Urban", "Rural"], help="Area location of the collateral property.")
        
    with col2:
        st.subheader("💰 Financial & Loan Metrics")
        
        applicant_income = st.number_input(
            "Applicant Monthly Income ($)", 
            min_value=0, 
            value=5000, 
            step=500, 
            help="Primary monthly income of the main applicant."
        )
        
        coapplicant_income = st.number_input(
            "Coapplicant Monthly Income ($)", 
            min_value=0.0, 
            value=0.0, 
            step=500.0, 
            help="Monthly income of the coapplicant, if applicable."
        )
        
        loan_amount = st.number_input(
            "Loan Amount (in Thousands, e.g., 150 = $150,000)", 
            min_value=1, 
            value=150, 
            step=10, 
            help="The requested loan limit."
        )
        
        loan_term = st.number_input(
            "Loan Term (in Months)", 
            min_value=12, 
            value=360, 
            step=12, 
            help="Repayment duration in months (e.g. 360 is typical for a 30-year mortgage)."
        )
        
        credit_history = st.selectbox(
            "Credit History Guidelines Met", 
            options=[1.0, 0.0], 
            format_func=lambda x: "Yes (1.0)" if x == 1.0 else "No (0.0)",
            help="Determines whether the customer meets standard credit rating guidelines."
        )

    st.write(" ")
    # Predict Button inside the form
    submit_button = st.form_submit_button(label="🎯 Run Approval Diagnosis")

# Inference Actions
if submit_button:
    if not pipeline_loaded:
        st.error("Cannot execute predictions. Please supply the pipeline artifact 'loan_approval_pipeline.pkl' to run the evaluation.")
    else:
        # Construct input DataFrame matching raw feature schema
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Married': [married],
            'Dependents': [str(dependents)],
            'Education': [education],
            'Self_Employed': [self_employed],
            'ApplicantIncome': [int(applicant_income)],
            'CoapplicantIncome': [float(coapplicant_income)],
            'LoanAmount': [int(loan_amount)],
            'Loan_Amount_Term': [int(loan_term)],
            'Credit_History': [float(credit_history)],
            'Property_Area': [property_area]
        })
        
        try:
            # Execute classification inference pipeline
            prediction = pipeline.predict(input_data)[0]
            probabilities = pipeline.predict_proba(input_data)[0]
            
            # Positive class probability (index 1 is approved status)
            prob_approved = probabilities[1] * 100
            prob_rejected = probabilities[0] * 100
            
            st.write("---")
            st.subheader("📊 Diagnostic Output Results")
            
            # Metric Card columns
            m_col1, m_col2 = st.columns(2)
            
            with m_col1:
                if prediction == 1 or prediction == "Y" or str(prediction).strip().upper() == "Y":
                    st.success("### 🎉 Approved!")
                    st.write("Applicant matches credit score criteria profiles.")
                else:
                    st.error("### ❌ Rejected")
                    st.write("Risk threshold exceeded or poor credit profile matching.")
            
            with m_col2:
                st.metric(label="System Approval Confidence Level", value=f"{prob_approved:.1f}%")
                st.progress(prob_approved / 100.0)
                
            # Informational Underwriting Context Cards
            st.info("""
            💡 **Underwriting Guidelines Checkpoint:**
            - **Credit History** is heavily weighted in the risk assessment process. A standard value of `1.0` acts as a primary decision driver.
            - **Income-to-Loan Proportion:** The ratio between total applicant income and the requested loan amount must support stable monthly amortization payments.
            """)
            
        except Exception as err:
            st.error("An error occurred during pipeline inference execution.")
            st.code(str(err))
