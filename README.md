# 📱 Telecom Customer Churn Prediction System

A machine learning system that predicts customer churn 
for telecom companies with explainability and 
personalized retention strategies.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.46-red)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-green)
![SHAP](https://img.shields.io/badge/SHAP-0.51-orange)

## 🔗 Live Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

## 🎯 Project Overview

This system helps telecom companies:
- **Predict** which customers will churn
- **Explain** WHY they will churn (SHAP)
- **Prioritize** customers by risk level
- **Suggest** personalized retention strategies

## 📊 Key Results

| Metric | Score |
|--------|-------|
| Accuracy | 94.01% |
| AUC-ROC | 93.81% |
| Precision | 73.54% |
| Recall | 64.38% |
| F1-Score | 68.66% |

## 🎯 Risk Segmentation

| Category | Customers | Actual Churn Rate |
|----------|-----------|-------------------|
| 🔴 RED (Urgent) | 7,396 | 93.9% |
| 🟡 YELLOW (Monitor) | 3,279 | 61.8% |
| 🟢 GREEN (Safe) | 89,324 | 1.4% |

## ✨ Unique Features

1. **SHAP Explainability** 
   - Shows exactly WHY each customer will churn
   - Existing systems are black box

2. **THREE-TIER Risk System**
   - 🔴 RED = Call within 24 hours
   - 🟡 YELLOW = SMS within 1 week  
   - 🟢 GREEN = Monthly engagement

3. **Personalized Retention Strategy**
   - Different offer for each customer
   - Based on actual usage behavior

4. **Indian Telecom Specific**
   - 99,999 real Indian customers
   - ARPU in Indian Rupees (₹)

## 🏗️ Project Structure
Telecom_churn_project/
│
├── app.py ← Streamlit Web App
├── requirements.txt ← Dependencies
│
├── models/
│ ├── final_xgboost_model.pkl
│ ├── scaler.pkl
│ ├── shap_explainer.pkl
│ └── feature_names.json
│
├── results/
│ ├── model_results.csv
│ ├── project_summary.json
│ ├── shap_importance.csv
│ └── best_params.json
│
└── assets/
├── final_dashboard.png
├── confusion_roc.png
├── shap_summary.png
└── risk_segmentation.png


## 🔧 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.13 | Programming Language |
| XGBoost | Final ML Model |
| SHAP | Model Explainability |
| Streamlit | Web Application |
| Scikit-learn | ML Utilities |
| SMOTE | Class Imbalance |
| Plotly | Interactive Charts |
| Google Colab | Model Training |

## 📋 Methodology
Raw Data (99,999 × 226)
↓
Data Preprocessing
↓
Feature Engineering (22 new features)
↓
Train-Test Split (80/20) + SMOTE
↓
4 Models Trained
↓
XGBoost Hyperparameter Tuning
↓
SHAP Explainability
↓
Risk Segmentation (RED/YELLOW/GREEN)
↓
Retention Strategy Engine
↓
Streamlit Web App