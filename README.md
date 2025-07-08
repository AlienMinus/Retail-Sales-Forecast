# 🛍️ Retail Sales Forecasting Dashboard

An advanced, interactive web application for **Retail Sales Forecasting**, built using **Python** and **Streamlit**. This project combines data visualization, machine learning, and model interpretability to provide actionable insights for retail business decision-making.

## 📊 Overview

This dashboard is designed to empower retail businesses by:
- Visualizing historical sales trends.
- Monitoring real-time KPIs.
- Forecasting future sales using an ML model (XGBoost).
- Explaining model predictions using SHAP.
- Allowing model retraining with new data.

It’s a **multi-page Streamlit app** with powerful backend integration for data preprocessing, visualization, forecasting, and continuous learning.

---

## 🎯 Objectives

- 📈 Comprehensive Sales Visualization  
- 🧮 KPI Monitoring  
- 🧠 ML-Based Forecasting (XGBoost)  
- 🔍 SHAP Interpretability  
- 🧩 Dynamic Filtering by Store, Department, Season, etc.  
- 🔁 Upload & Retrain Model on New Data  
- 💻 User-Friendly Interface for Analysts, Managers & Data Scientists

---

## 🛠️ Tech Stack

| Component     | Technology                     |
|--------------|---------------------------------|
| Language      | Python                         |
| Framework     | Streamlit                      |
| ML Algorithm  | XGBoost Regressor              |
| Visualizations| Plotly, Seaborn, Matplotlib    |
| ML Libraries  | scikit-learn, SHAP             |
| Others        | Pandas, NumPy, Joblib, datetime|

---

## 📁 Architecture

Retail-Sales-Forecasting/
├── app.py # Main launcher (Streamlit entry point)
├── dataset/
│ └── dataset.csv # Historical retail sales data
├── pages/
│ ├── 1_Dashboard.py # Sales analysis & KPIs
│ ├── 2_Forecasting.py # Model evaluation & forecasting
│ └── 3_Retrain_Model.py # Upload new data & retrain model
├── model/
│ └── xgboost_model.pkl # Serialized trained model
└── README.md

---

## 📉 Key Features

### 🔹 Interactive Dashboard
- Filter by store, department, type, season, holiday, etc.
- Line, bar, pie, and box plots
- Real-time KPIs: total sales, avg. weekly sales, store/department count
- Correlation heatmap and data summaries

### 🔹 Advanced Forecasting
- XGBoost regression model
- R² Score, MAE, RMSE evaluation
- Predict weekly sales for user-selected store & department
- Actual vs. Predicted sales plot

### 🔹 SHAP Model Interpretability
- Feature importance via SHAP bar plot and beeswarm
- Understand *why* certain sales are predicted

### 🔹 Model Retraining
- Upload new CSV data
- Auto-cleaning, merging, and retraining of the model
- Re-deployment with updated predictions

---

## 🧪 How to Run

1. **Clone this repo:**
   ```bash
   git clone https://github.com/<your-username>/Retail-Sales-Forecasting.git
   cd Retail-Sales-Forecasting
   ```
2. **Create a virtual environment and activate it:**
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```
3. **Install required dependencies:**
  ```bash
  pip install -r requirements.txt
  ```
4. **Launch the Streamlit app:**
  ```bash
  streamlit run app.py
  ```
## 📈 Sample Dataset
The dataset contains:
- Date, Store, Dept, Weekly_Sales, Temperature, Fuel_Price
- MarkDown1-5, CPI, Unemployment, IsHoliday
- Feature engineered columns like Month, Year, Season

## 📦 Outputs
- 🌐 Multi-page Streamlit app
- 📊 KPI & trend dashboards
- 📈 Forecast plots and downloadable tables
- 🔁 Retraining confirmation messages

## 💡 Future Scope
- 📈 Add LSTM, Prophet, ARIMA models for benchmarking
- 🌍 API Integration (e.g., real-time weather, economic data)
- 🧾 Authentication & Role-based Access
- 💾 SQL/NoSQL database integration
- 🔁 CI/CD for automated model retraining & deployment
- 📉 What-If Analysis & Forecast Uncertainty
- 🔍 Anomaly detection in historical trends

## 📚 License
MIT License. See LICENSE file for more details.

## 🤝 Acknowledgements
- Supervised by Mr. Sambit Subhasish Sahu
- Developed as part of a Machine Learning Project (June 2025)
- By Manas R. Das
- 📧 dasmanasranjan2005@gmail.com

> "Our greatest glory is not in never failing, but in rising every time we fail." — Confucius

