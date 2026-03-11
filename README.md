# 📊 Telco Customer Churn Prediction
*An end-to-end Machine Learning project focusing on customer retention, model interpretability, and production-ready deployment.*

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![CatBoost](https://img.shields.io/badge/Model-CatBoost-ff5a1f?style=flat)](https://catboost.ai/)
[![SHAP](https://img.shields.io/badge/Interpretability-SHAP-00aff0?style=flat)](https://shap.readthedocs.io/)

## 📌 Project Overview
The goal of this project is to predict customer churn for a telecommunications company and identify the key drivers behind customer attrition. Using the **IBM Telco Churn dataset**, I developed a high-performance classification model that provides actionable business insights through model interpretability.

## 🚀 Key Features
- **Native Categorical Support:** Utilized **CatBoost** to handle categorical features efficiently without manual one-hot encoding.
- **Explainable AI (XAI):** Integrated **SHAP** to move beyond "black-box" predictions and visualize feature importance.
- **Robust Pipeline:** Built a scalable workflow from raw data ingestion to model persistence.

## 📂 Repository Structure
- `notebooks/`: Comprehensive Jupyter Notebook containing EDA, Feature Engineering, and Model Training.
- `models/`: Serialized trained model (`final_churn_model.joblib`) ready for deployment.
- `app/`: (In Progress) Streamlit web application.
- `docker/`: (In Progress) Dockerization files for portable deployment.

## 🔍 Model Interpretability (SHAP Analysis)
One of the core strengths of this project is understanding **why** a customer might leave. 

### Global Importance
According to the SHAP analysis, the top 3 drivers for churn are:
1. **Contract Type:** Customers on "Month-to-month" plans have a significantly higher risk of churn.
2. **Internet Service:** Fiber optic users show higher attrition rates compared to DSL users.
3. **Tenure & Monthly Charges:** The relationship between loyalty and cost is a critical factor.



## 🛠️ How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Telco-Churn-Prediction.git](https://github.com/YOUR_USERNAME/Telco-Churn-Prediction.git)