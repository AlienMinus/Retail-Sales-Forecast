# app.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from datetime import timedelta

# --- Page Setup ---
st.set_page_config(layout="wide", page_title="Advanced Sales Dashboard")
st.title("📊 Advanced Retail Sales Data Dashboard")
st.markdown("Dive deep into your sales data, visualize trends, and forecast future performance with powerful machine learning insights.")

# --- Helper Functions (defined globally in app.py and stored in session_state) ---
def get_season(month):
    return (
        'Winter' if month in [12, 1, 2] else
        'Spring' if month in [3, 4, 5] else
        'Summer' if month in [6, 7, 8] else
        'Fall'
    )

# Store get_season in session state for access by other pages
if 'get_season' not in st.session_state:
    st.session_state.get_season = get_season
    st.info("Helper function 'get_season' initialized in session state.")

# --- Global Encoders (initialized and stored in session_state) ---
if 'type_encoder' not in st.session_state:
    st.session_state.type_encoder = LabelEncoder()
    st.info("type_encoder initialized in session state.")
if 'season_encoder' not in st.session_state:
    st.session_state.season_encoder = LabelEncoder()
    st.info("season_encoder initialized in session state.")

# --- Model Path (global) ---
if 'model_path' not in st.session_state:
    st.session_state.model_path = 'model/xgboost_model.pkl'
    os.makedirs('model', exist_ok=True) # Ensure model directory exists
    st.info(f"Model path set to: {st.session_state.model_path}")

@st.cache_data
def load_and_preprocess_data():
    st.info("Loading and preprocessing data...")
    try:
        df = pd.read_csv("dataset/dataset.csv")
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
        df['Season'] = df['Date'].dt.month.map(st.session_state.get_season) # Use get_season from session state

        type_encoder = st.session_state.type_encoder
        season_encoder = st.session_state.season_encoder

        # Fit encoders on all unique values from the original data
        type_encoder.fit(df['Type'].unique())
        season_encoder.fit(df['Season'].unique())

        df['Type_Encoded'] = type_encoder.transform(df['Type'])
        df['Season_Encoded'] = season_encoder.transform(df['Season'])
        df['IsHoliday'] = df['IsHoliday'].astype(int)
        st.success("Data loaded and preprocessed successfully.")
        return df
    except FileNotFoundError:
        st.error("Error: dataset/dataset.csv not found. Please ensure the file exists in the correct path.")
        st.stop() # Stop the app if data file is missing
    except Exception as e:
        st.error(f"Error during data loading or preprocessing: {e}")
        st.stop() # Stop the app on other data errors

# Load dataset once and store in session state
if 'df' not in st.session_state:
    st.session_state.df = load_and_preprocess_data()
    if st.session_state.df is not None:
        st.success(f"DataFrame loaded into session state with {len(st.session_state.df)} rows.")
    else:
        st.error("DataFrame failed to load and is None in session state.")

# Initialize xgboost_model in session state
if 'xgboost_model' not in st.session_state:
    st.session_state.xgboost_model = None
    st.info("XGBoost model placeholder initialized in session state.")
    if os.path.exists(st.session_state.model_path):
        st.info("Attempting to load pre-trained model from disk...")
        try:
            # Check if model can be loaded immediately
            temp_model = joblib.load(st.session_state.model_path)
            # You might want a more robust feature check here, but for initial load, just loading is fine.
            st.session_state.xgboost_model = temp_model
            st.success("Pre-trained XGBoost model loaded into session state.")
        except Exception as e:
            st.warning(f"Could not load pre-trained model '{st.session_state.model_path}': {e}. Model will be trained on Forecasting page if needed.")


st.markdown("""
---
Welcome to the Advanced Retail Sales Dashboard!
Use the sidebar to navigate through different sections:
- **Sales Dashboard:** Explore key sales metrics, trends, and distributions.
- **Sales Forecasting:** Predict future sales for specific stores and departments.
- **Retrain Model:** Upload new data to retrain the underlying machine learning model.
""")

# --- Footer (optional, could be in app.py or each page) ---
st.markdown("---")
st.markdown("Made with ❤️ by AlienMinus | Powered by Streamlit & Plotly")