# hr_salary_prediction

\\# Advanced Salary Prediction \\\& Pay Band Classification System

\\## Overview

This project presents a complete end-to-end Machine Learning pipeline for predicting expected CTC (Cost to Company) based on candidate attributes such as experience, role, department, education, certifications, and industry.

In addition to regression-based salary prediction, the system also performs salary band classification, enabling structured pay-band mapping for HR decision-making.
---

\\## Problem Statement

Accurate salary estimation is a critical challenge in recruitment and workforce planning. Traditional rule-based systems fail to capture nonlinear relationships between candidate features and compensation.

============

This project builds a data-driven predictive system to:


\\- Estimate expected CTC using regression models

\\- Classify candidates into structured pay bands

\\- Provide interpretable performance evaluation

---

\\## Dataset Features


\\- Total Experience

\\- Field Experience

\\- Department

\\- Role

\\- Industry

\\- Education

\\- Graduation Specialization

\\- Location Preferences

\\- Certifications

\\- International Degree Indicator


---


\\## Technical Architecture


1\\. Data Cleaning \\\& Validation

2\\. Feature Engineering

3\\. Categorical Encoding (OneHotEncoder)

4\\. Train-Test Split

5\\. Model Training (Multiple Algorithms)

6\\. Cross-Validation \\\& Hyperparameter Tuning

7\\. Model Evaluation

8\\. Model Serialization (.pkl files)



---



\\## Models Implemented


\\- Linear Regression

\\- Decision Tree Regressor

\\- Random Forest Regressor

\\- HistGradientBoosting Regressor

\\- XGBoost Regressor


Best model selected based on R² score and generalization performance.



---

\\## Evaluation Metrics


\\- R² Score

\\- Mean Absolute Error (MAE)

\\- Mean Squared Error (MSE)


This ensures both accuracy and robustness of predictions.



---



\\## Key Achievements


\\- Built a production-ready ML pipeline

\\- Implemented multi-model comparison

\\- Applied structured validation checks

\\- Saved deployable serialized models

\\- Created separate prediction module (predict.py)

\\- Generated EDA visualizations for interpretability


---


\\## Project Structure


\\- Train\\\_model.py → End-to-end model training pipeline

\\- predict.py → Prediction logic for new candidates

\\- expected\\\_ctc.csv → Training dataset

\\- \\\*.pkl → Serialized trained models

\\- \\\*.png → EDA visualizations


---


\\## Conclusion


This project demonstrates practical implementation of supervised machine learning for real-world salary prediction and pay-band structuring, combining statistical rigor, modular architecture, and deployment-ready design.
