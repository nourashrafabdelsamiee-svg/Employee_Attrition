# Employee Attrition Prediction – End-to-End ML Project

# Overview
This project builds a complete Employee Attrition Prediction system using machine learning. It covers the full workflow starting from data exploration and preprocessing, through model development, and ending with deployment using Streamlit. The objective is to help HR teams identify employees at high risk of leaving the organization.

# Key Features
Exploratory Data Analysis (EDA)
Handling class imbalance
Feature engineering
Preprocessing pipeline (encoding, scaling)
Model development and comparison
Saving trained model and preprocessing pipeline
Deployment using Streamlit with manual entry, CSV upload, and interactive dashboard

# Data Exploration
The dataset was analyzed to identify key patterns related to employee attrition.
Insights included:
Clear class imbalance requiring resampling
Significant relations with attrition such as overtime, job satisfaction, tenure, income and job level
Categorical variables requiring encoding

# Preprocessing
Numerical features: scaling and handling skewed distributions

Categorical features: one-hot encoding
Missing value handling
Additional engineered features such as tenure categories and salary bands

# Model Development
Multiple models were trained and evaluated.
XGBoost achieved the best performance compared to other models including Random Forest and Logistic Regression.

Evaluation metrics included:
Accuracy
ROC-AUC
Precision and Recall
Confusion Matrix

# Deployment (Streamlit)
The Streamlit app includes:
Manual data entry form for single employee prediction
CSV upload for batch predictions
Interactive dashboard built with Plotly showing:
Attrition probability distribution
Attrition by department
Attrition by marital status
Attrition by overtime

## Project Structure
employee-attrition-prediction/
├── streamlit_app.py        # Main Streamlit application
├── xgboost_model.pkl       # Trained & tuned XGBoost model
├── preprocessor.pkl        # Full preprocessing pipeline
├── project_depi.ipynb      # Complete Jupyter notebook
└── README.md               # Project description


## How to Run Locally
```bash
git clone (https://github.com/nourashrafabdelsamiee-svg/Employee_Attrition)
cd employee-attrition-prediction
streamlit run streamlit_app.py

