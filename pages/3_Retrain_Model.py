# pages/3_Retrain_Model.py
import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
import joblib
import os

# Access global data and encoders from st.session_state
df = st.session_state.df
type_encoder = st.session_state.type_encoder
season_encoder = st.session_state.season_encoder
model_path = st.session_state.model_path
get_season = st.session_state.get_season # Access the helper function

st.header("🔁 Retrain Model with New Data")
st.markdown("Upload new sales data to retrain the XGBoost model and improve its accuracy.")

uploaded_file = st.file_uploader("Upload new sales CSV data", type=["csv"])
if uploaded_file:
    try:
        new_data_df = pd.read_csv(uploaded_file)
        st.info("Processing new data...")

        raw_required_cols = [
            'Year', 'Month', 'Day', 'Store', 'Dept', 'Type', 'Size',
            'IsHoliday', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2',
            'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment', 'Weekly_Sales'
        ]
        if not all(col in new_data_df.columns for col in raw_required_cols):
            missing = [col for col in raw_required_cols if col not in new_data_df.columns]
            st.error(f"Missing required columns in uploaded file: {', '.join(missing)}. Please ensure your CSV has all necessary columns.")
            st.stop()

        new_data_df['Date'] = pd.to_datetime(new_data_df[['Year', 'Month', 'Day']])
        new_data_df['Season'] = new_data_df['Date'].dt.month.map(get_season)

        # Update encoders with new data categories
        all_types_for_fit = list(type_encoder.classes_) + list(new_data_df['Type'].unique())
        type_encoder.fit(sorted(list(set(all_types_for_fit))))
        new_data_df['Type_Encoded'] = type_encoder.transform(new_data_df['Type'])

        all_seasons_for_fit = list(season_encoder.classes_) + list(new_data_df['Season'].unique())
        season_encoder.fit(sorted(list(set(all_seasons_for_fit))))
        new_data_df['Season_Encoded'] = season_encoder.transform(new_data_df['Season'])

        new_data_df['IsHoliday'] = new_data_df['IsHoliday'].astype(int)

        # Combine original processed ML data (`df` contains `_Encoded` columns from `load_and_preprocess_data`) with new data
        combined_ml_df = pd.concat([df, new_data_df], ignore_index=True)

        # Drop duplicates based on identifying columns (Date, Store, Dept)
        combined_ml_df.drop_duplicates(subset=['Date', 'Store', 'Dept'], inplace=True)

        st.info(f"Combined historical data with {len(new_data_df)} new records. Total records for retraining: {len(combined_ml_df)}")

        features = [
            'Day', 'Month', 'Year', 'Store', 'Dept', 'Type_Encoded', 'Size',
            'IsHoliday', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2',
            'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment', 'Season_Encoded'
        ]
        X_combined = combined_ml_df[features]
        y_combined = combined_ml_df['Weekly_Sales']

        st.info("Retraining XGBoost model with combined dataset...")
        retrained_model = XGBRegressor(objective='reg:squarederror', n_estimators=200, random_state=42,
                                       learning_rate=0.1, max_depth=5, subsample=0.7, colsample_bytree=0.7)
        retrained_model.fit(X_combined, y_combined)
        joblib.dump(retrained_model, model_path)
        st.success("XGBoost model successfully retrained and saved!")
        st.info("Reloading dashboard to reflect updated model and data. This may take a moment.")

        # Update session state with the new data and model before rerunning
        st.session_state.df = combined_ml_df
        st.session_state.type_encoder = type_encoder
        st.session_state.season_encoder = season_encoder
        st.session_state.xgboost_model = retrained_model # Update the model in session state

        st.experimental_rerun()

    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
        st.warning("Please ensure your CSV file is correctly formatted and contains all necessary columns.")
        st.exception(e)

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ by AlienMinus | Powered by Streamlit & Plotly")