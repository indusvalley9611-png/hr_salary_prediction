import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# SALARY PREDICTION - MODEL TRAINING PIPELINE
# =========================================================

# =========================================================
# Load Data
# =========================================================
df = pd.read_csv("expected_ctc.csv")
initial_rows = df.shape[0]

print("\nSalary Prediction - Model Training Pipeline")
print("Original dataset shape:", df.shape)


# =========================================================
# Data Cleaning
# =========================================================
print("\nStarting data cleaning...")

df = df.drop_duplicates()

text_cols = df.select_dtypes(include=["object", "string"]).columns
for col in text_cols:
    df[col] = df[col].astype(str).str.strip().str.title()

df = df[df["Total_Experience"] >= 0]
df = df[df["Total_Experience_in_field_applied"] >= 0]
df = df[df["Total_Experience_in_field_applied"] <= df["Total_Experience"]]
df = df[df["Total_Experience"] <= 40]

q_low = df["Expected_CTC"].quantile(0.01)
q_high = df["Expected_CTC"].quantile(0.995)

df = df[
    (df["Expected_CTC"] >= q_low) &
    (df["Expected_CTC"] <= q_high)
]

print("Dataset shape after cleaning:", df.shape)
print("Rows removed:", initial_rows - df.shape[0])


# =========================================================
# Exploratory Data Analysis
# =========================================================
print("\nRunning exploratory data analysis...")

plt.figure(figsize=(8, 5))
sns.histplot(df["Expected_CTC"], bins=50, kde=True)
plt.title("Salary Distribution")
plt.tight_layout()
plt.savefig("eda_salary_distribution.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(
    x="Total_Experience",
    y="Expected_CTC",
    data=df,
    alpha=0.3
)
plt.title("Total Experience vs Salary")
plt.tight_layout()
plt.savefig("eda_experience_vs_salary.png")
plt.close()

numeric_df = df.select_dtypes(include=["int64", "float64"])
corr = numeric_df.corr()["Expected_CTC"].sort_values(ascending=False)

plt.figure(figsize=(6, 5))
sns.barplot(x=corr.values, y=corr.index)
plt.title("Correlation with Salary")
plt.tight_layout()
plt.savefig("eda_salary_correlation.png")
plt.close()

print("EDA completed.")


# =========================================================
# Feature Engineering
# =========================================================
df["Experience_Level"] = pd.cut(
    df["Total_Experience"],
    bins=[0, 2, 5, 8, 12, 18, 25, 40],
    labels=["Fresher", "Junior", "Mid", "Senior", "Lead", "Expert", "Veteran"]
)

print("Feature engineering completed.")


# =========================================================
# Train-Test Split (Anti-Leakage)
# =========================================================
features = [
    "Total_Experience",
    "Total_Experience_in_field_applied",
    "Experience_Level",
    "Department",
    "Role",
    "Industry",
    "Education",
    "Graduation_Specialization",
    "Current_Location",
    "Preferred_location",
    "No_Of_Companies_worked",
    "Certifications",
    "International_degree_any"
]

X = df[features]
y = df["Expected_CTC"]

X_train, X_test, y_train_raw, y_test_raw = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

y_train = np.log1p(y_train_raw)
y_test = np.log1p(y_test_raw)


# =========================================================
# Preprocessing
# =========================================================
cat_cols = X_train.select_dtypes(include=["object", "string", "category"]).columns
num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])


# =========================================================
# Base Model Benchmarking
# =========================================================
print("\nModel benchmarking...")

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, n_jobs=-1)
}

results = []

for name, model in models.items():

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    cv_scores = cross_val_score(
        pipe,
        X_train,
        y_train,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    print(name)
    print("CV Mean R2:", round(cv_scores.mean(), 4))
    print("CV Std Dev:", round(cv_scores.std(), 4))
    print()

    results.append({
        "Model": name,
        "Estimator": model,
        "CV_Mean": cv_scores.mean()
    })


# =========================================================
# Select Top 2 Models
# =========================================================
results = sorted(results, key=lambda x: x["CV_Mean"], reverse=True)
top_models = results[:2]

print("Top models selected for tuning:")
for m in top_models:
    print(m["Model"])


# =========================================================
# Hyperparameter Tuning
# =========================================================
best_model = None
best_score = -np.inf

for entry in top_models:

    if entry["Model"] == "Random Forest":
        param_dist = {
            "regressor__n_estimators": [300, 400, 500],
            "regressor__max_depth": [6, 8, 10],
            "regressor__min_samples_split": [10, 20, 30],
            "regressor__min_samples_leaf": [8, 12, 20],
            "regressor__max_features": ["sqrt", 0.7]
        }

    elif entry["Model"] == "HistGradientBoosting":
        param_dist = {
            "regressor__max_depth": [4, 6],
            "regressor__learning_rate": [0.03],
            "regressor__max_iter": [300],
            "regressor__min_samples_leaf": [40, 60, 80],
            "regressor__l2_regularization": [2.0, 5.0, 10.0]
        }

    else:
        param_dist = {
            "regressor__max_depth": [4, 6],
            "regressor__learning_rate": [0.03, 0.05],
            "regressor__n_estimators": [200, 300]
        }

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", entry["Estimator"])
    ])

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=10,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    search.fit(X_train, y_train)

    if search.best_score_ > best_score:
        best_score = search.best_score_
        best_model = search.best_estimator_


# =========================================================
# Final Evaluation
# =========================================================
pred_log = best_model.predict(X_test)
pred_salary = np.expm1(pred_log)

r2 = r2_score(y_test_raw, pred_salary)
mae = mean_absolute_error(y_test_raw, pred_salary)
rmse = np.sqrt(mean_squared_error(y_test_raw, pred_salary))

print("\nFinal model performance")
print("Selected model:", type(best_model.named_steps["regressor"]).__name__)
print("Test R2:", round(r2, 4))
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))

train_pred_log = best_model.predict(X_train)
train_pred = np.expm1(train_pred_log)

# =========================================================
# Create Pay Bands
# =========================================================
_, bin_edges = pd.qcut(
    df["Expected_CTC"],
    q=6,
    retbins=True,
    duplicates="drop"
)
# =========================================================
# Save Artifacts
# =========================================================
joblib.dump(best_model, "salary_model.pkl")
joblib.dump(bin_edges, "salary_bin_edges.pkl")
joblib.dump(df["Total_Experience"].max(), "max_experience.pkl")

print("Artifacts saved successfully...")