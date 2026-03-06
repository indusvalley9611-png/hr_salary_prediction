import pandas as pd
import numpy as np
import joblib


# =========================================================
# SALARY PREDICTION - INFERENCE PIPELINE
# =========================================================


# =========================================================
# Load Saved Artifacts
# =========================================================
model = joblib.load("salary_model.pkl")
bin_edges = joblib.load("salary_bin_edges.pkl")
max_experience = joblib.load("max_experience.pkl")


# =========================================================
# Candidate Input
# =========================================================
candidate_dict = {
    "Total_Experience": 10,
    "Total_Experience_in_field_applied": 7,
    "Experience_Level": "Mid",
    "Department": "Education",
    "Role": "Senior Analyst",
    "Industry": "Finance",
    "Education": "Pg",
    "Graduation_Specialization": "Mathematics",
    "Current_Location": "Mumbai",
    "Preferred_location": "Mumbai",
    "No_Of_Companies_worked": 3,
    "Certifications": 2,
    "International_degree_any": 0
}


# =========================================================
# Input Validation
# =========================================================
if candidate_dict["Total_Experience"] < 0:
    print("Invalid input: Total experience cannot be negative.")
    exit()

if candidate_dict["Total_Experience_in_field_applied"] < 0:
    print("Invalid input: Field experience cannot be negative.")
    exit()

if candidate_dict["Total_Experience_in_field_applied"] > candidate_dict["Total_Experience"]:
    print("Invalid input: Field experience cannot exceed total experience.")
    exit()


# =========================================================
# Create DataFrame
# =========================================================
candidate = pd.DataFrame([candidate_dict])

# Clip unrealistic experience beyond training data
candidate["Total_Experience"] = candidate["Total_Experience"].clip(
    upper=max_experience
)


# =========================================================
# Recreate Feature Engineering
# =========================================================
candidate["Experience_Level"] = pd.cut(
    candidate["Total_Experience"],
    bins=[0, 2, 5, 8, 12, 18, 25, 40],
    labels=["Fresher", "Junior", "Mid", "Senior", "Lead", "Expert", "Veteran"]
)


# =========================================================
# Ensure Correct Column Order
# =========================================================
expected_features = model.named_steps["preprocessor"].feature_names_in_
candidate = candidate[expected_features]


# =========================================================
# Predict Salary
# =========================================================
pred_log = model.predict(candidate)[0]
pred_salary = np.expm1(pred_log)


# =========================================================
# Enforce Realistic Salary Bounds
# =========================================================
min_salary = bin_edges[0]
max_salary = bin_edges[-1]

if pred_salary < min_salary:
    pred_salary = min_salary

if pred_salary > max_salary:
    pred_salary = max_salary


# =========================================================
# Determine Pay Band Range
# =========================================================
labels = [
    f"Pay_Band_{i+1}"
    for i in range(len(bin_edges) - 1)
]

pred_band = pd.cut(
    [pred_salary],
    bins=bin_edges,
    labels=labels,
    include_lowest=True
)[0]

band_index = labels.index(pred_band)
range_min = bin_edges[band_index]
range_max = bin_edges[band_index + 1]


# =========================================================
# Final Output
# =========================================================
print("\nSalary Recommendation")
print(
    "Recommended Range: {:.2f} LPA - {:.2f} LPA".format(
        range_min / 100000,
        range_max / 100000
    )
)
print("Pay Band:", pred_band)