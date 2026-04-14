import streamlit as st
import pandas as pd
import joblib
import numpy as np

# LOAD ARTIFACTS
model = joblib.load("salary_model.pkl")
bin_edges = joblib.load("salary_bin_edges.pkl")
max_exp = joblib.load("max_experience.pkl")

st.set_page_config(page_title="Salary Predictor", layout="wide")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("💼 Salary Prediction App")
st.sidebar.info("Predict expected CTC based on profile")

page = st.sidebar.radio("Navigation", ["Predict Salary", "About"])

# =========================
# MAIN PAGE
# =========================
if page == "Predict Salary":

    st.title("💰 Expected Salary Prediction")

    st.markdown("### Enter Candidate Details")

    col1, col2 = st.columns(2)

    with col1:
        total_exp = st.slider("Total Experience", 0, int(max_exp), 2)
        field_exp = st.slider("Experience in Field", 0, int(total_exp), 1)
        companies = st.number_input("Companies Worked", 0, 20, 1)
        certs = st.number_input("Certifications", 0, 20, 0)

        dept = st.selectbox("Department", ["IT", "Finance", "HR", "Sales", "Other"])
        role = st.text_input("Role", "Software Engineer")

    with col2:
        industry = st.selectbox("Industry", ["Tech", "Banking", "Healthcare", "Other"])
        education = st.selectbox("Education", ["Bachelors", "Masters", "PhD"])
        specialization = st.text_input("Specialization", "Computer Science")

        location = st.selectbox("Current Location", ["Bangalore", "Mumbai", "Delhi", "Other"])
        pref_location = st.selectbox("Preferred Location", ["Bangalore", "Mumbai", "Delhi", "Other"])

        intl_degree = st.selectbox("International Degree", ["Yes", "No"])

    # =========================
    # FEATURE ENGINEERING MATCH
    # =========================
    def get_exp_level(exp):
        if exp <= 2: return "Fresher"
        elif exp <= 5: return "Junior"
        elif exp <= 8: return "Mid"
        elif exp <= 12: return "Senior"
        elif exp <= 18: return "Lead"
        elif exp <= 25: return "Expert"
        else: return "Veteran"

    exp_level = get_exp_level(total_exp)

    # =========================
    # PREDICTION
    # =========================
    if st.button("🚀 Predict Salary", use_container_width=True):

        input_df = pd.DataFrame([{
            "Total_Experience": total_exp,
            "Total_Experience_in_field_applied": field_exp,
            "Experience_Level": exp_level,
            "Department": dept,
            "Role": role,
            "Industry": industry,
            "Education": education,
            "Graduation_Specialization": specialization,
            "Current_Location": location,
            "Preferred_location": pref_location,
            "No_Of_Companies_worked": companies,
            "Certifications": certs,
            "International_degree_any": intl_degree
        }])

        pred_log = model.predict(input_df)
        salary = np.expm1(pred_log)[0]

        # =========================
        # PAY BAND
        # =========================
        band = np.digitize(salary, bin_edges)

        # =========================
        # OUTPUT
        # =========================
        st.success("Prediction Complete!")

        c1, c2 = st.columns(2)

        with c1:
            st.metric("💰 Estimated Salary (CTC)", f"₹ {salary:,.0f}")

        with c2:
            st.metric("📊 Pay Band", f"Level {band}")

        st.progress(min(salary / (max(bin_edges)), 1.0))


# =========================
# ABOUT PAGE
# =========================
elif page == "About":
    st.title("📘 About This App")

    st.write("""
    This app predicts expected salary (CTC) using:
    
    - Machine Learning Pipeline
    - Feature Engineering
    - Hyperparameter Tuning
    - Log Transformation
    
    Built using:
    - Scikit-learn
    - XGBoost
    - Streamlit
    
    """)
