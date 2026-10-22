import os
import pickle

import joblib
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stButton>button, .stFormSubmitButton>button {
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
    .stButton>button:hover, .stFormSubmitButton>button:hover {
        background-color: #17b978;
        color: white;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Custom transformers used inside the saved pipeline.
# These MUST match the notebook exactly, otherwise loading the .pkl fails with
# "Can't get attribute 'FeatureEngineer' on <module '__main__'>".
# ----------------------------------------------------------------------------
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        X_out["TotalIncome"] = X_out["ApplicantIncome"] + X_out["CoapplicantIncome"]
        X_out["EMI"] = X_out["LoanAmount"] / X_out["Loan_Amount_Term"].replace(0, 360)
        X_out["Balance_Income"] = X_out["TotalIncome"] - (X_out["EMI"] * 1000)
        X_out["Log_TotalIncome"] = np.log1p(X_out["TotalIncome"])
        X_out["Log_LoanAmount"] = np.log1p(X_out["LoanAmount"])
        return X_out


class DataFrameReconstructor(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols, cat_cols):
        self.num_cols = num_cols
        self.cat_cols = cat_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        col_names = self.num_cols + self.cat_cols
        df_reconstructed = pd.DataFrame(X, columns=col_names)
        for col in self.num_cols:
            df_reconstructed[col] = pd.to_numeric(df_reconstructed[col])
        return df_reconstructed

MODEL_FILE = "loan_approval_pipeline.pkl"


# ----------------------------------------------------------------------------
# Model loading (cached so it only loads once)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_pipeline(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"'{path}' not found next to app.py.")
    try:
        return joblib.load(path)          # file saved with joblib.dump
    except Exception:
        with open(path, "rb") as f:       # fallback: file saved with pickle
            return pickle.load(f)


pipeline = None
load_error = None
try:
    pipeline = load_pipeline(MODEL_FILE)
except Exception as e:
    load_error = e


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "This app predicts whether a loan application is likely to be "
        "**approved** using a machine learning pipeline trained on historical "
        "loan data (614 applications)."
    )
    st.markdown(
        "**How to use**\n"
        "1. Fill in the applicant details\n"
        "2. Click **Run Approval Diagnosis**\n"
        "3. Read the prediction and confidence"
    )
    st.divider()
    if pipeline is not None:
        st.success("Model loaded ✅")
    else:
        st.error("Model not loaded ❌")
    st.caption(
        "⚠️ Academic project. Predictions are for demonstration only and must "
        "not be used for real lending decisions."
    )

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🏦 Loan Approval Prediction")
st.markdown(
    "*Enter the applicant's details to estimate the probability of loan approval.*"
)

if load_error is not None:
    st.error("Model could not be loaded.")
    st.code(str(load_error))
    st.info(
        f"Make sure **{MODEL_FILE}** is in the same folder as `app.py`, and that "
        "the scikit-learn version in `requirements.txt` matches the one used "
        "in your Colab notebook (`import sklearn; print(sklearn.__version__)`)."
    )
    st.stop()

st.divider()

# ----------------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------------
with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Applicant Profile")
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        property_area = st.selectbox("Property Area", ["Semiurban", "Urban", "Rural"])

    with col2:
        st.subheader("💰 Financial Details")
        applicant_income = st.number_input(
            "Applicant Monthly Income", min_value=0, value=5000, step=500
        )
        coapplicant_income = st.number_input(
            "Coapplicant Monthly Income", min_value=0.0, value=0.0, step=500.0
        )
        loan_amount = st.number_input(
            "Loan Amount (in thousands, e.g. 150 = 150,000)",
            min_value=1, value=150, step=10,
        )
        loan_term = st.number_input(
            "Loan Term (months)", min_value=12, max_value=480, value=360, step=12
        )
        credit_history = st.selectbox(
            "Credit History meets guidelines?",
            options=[1.0, 0.0],
            format_func=lambda x: "Yes (1)" if x == 1.0 else "No (0)",
        )

    submitted = st.form_submit_button("🎯 Run Approval Diagnosis")

# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
if submitted:
    input_df = pd.DataFrame(
        {
            "Gender": [gender],
            "Married": [married],
            "Dependents": ["3" if dependents == "3+" else dependents],  # pipeline expects numeric
            "Education": [education],
            "Self_Employed": [self_employed],
            "ApplicantIncome": [int(applicant_income)],
            "CoapplicantIncome": [float(coapplicant_income)],
            "LoanAmount": [float(loan_amount)],
            "Loan_Amount_Term": [float(loan_term)],
            "Credit_History": [float(credit_history)],
            "Property_Area": [property_area],
        }
    )

    # Match the exact column order the pipeline was trained on (if available)
    expected = getattr(pipeline, "feature_names_in_", None)
    if expected is not None:
        missing = [c for c in expected if c not in input_df.columns]
        if missing:
            st.error(
                f"The model expects columns the app does not provide: {missing}. "
                "Either add them to the form or move that feature engineering "
                "inside the saved pipeline."
            )
            st.stop()
        input_df = input_df[list(expected)]

    try:
        pred = pipeline.predict(input_df)[0]
        proba = pipeline.predict_proba(input_df)[0]
        classes = list(getattr(pipeline, "classes_", [0, 1]))

        # Find the probability of the "approved" class (1 or "Y")
        approved_label = 1 if 1 in classes else "Y"
        prob_approved = float(proba[classes.index(approved_label)]) * 100
        approved = str(pred).strip().upper() in {"1", "Y", "TRUE"}

        st.divider()
        st.subheader("📊 Result")

        r1, r2 = st.columns(2)
        with r1:
            if approved:
                st.success("### 🎉 Likely Approved")
                st.write("The profile resembles historically approved applications.")
            else:
                st.error("### ❌ Likely Rejected")
                st.write("The profile resembles historically rejected applications.")
        with r2:
            st.metric("Approval probability", f"{prob_approved:.1f}%")
            st.progress(min(max(prob_approved / 100.0, 0.0), 1.0))

        # Simple, human-readable affordability context
        total_income = applicant_income + coapplicant_income
        emi = (loan_amount * 1000) / loan_term
        st.markdown("#### Affordability snapshot")
        a1, a2, a3 = st.columns(3)
        a1.metric("Total monthly income", f"{total_income:,.0f}")
        a2.metric("Approx. monthly EMI (no interest)", f"{emi:,.0f}")
        a3.metric("EMI / income", f"{(emi / total_income * 100) if total_income else 0:.0f}%")

        if credit_history == 0.0:
            st.warning(
                "Credit history is the strongest driver in this dataset. Applicants "
                "who do not meet credit guidelines were rarely approved."
            )

    except Exception as err:
        st.error("Prediction failed.")
        st.code(str(err))
        st.info(
            "This usually means a scikit-learn version mismatch or the pipeline "
            "expects different input columns than the form provides."
        )
