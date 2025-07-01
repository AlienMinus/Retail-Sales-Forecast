# pages/2_Forecasting.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split # Keep for evaluation section
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import shap
import joblib
import os
from datetime import timedelta
import matplotlib.pyplot as plt # For SHAP plots
from itertools import product # Import for vectorized future data generation

# --- Access global data and encoders from st.session_state ---
# These are guaranteed to be initialized by app.py (or the main script)
try:
    df = st.session_state.df
    type_encoder = st.session_state.type_encoder
    season_encoder = st.session_state.season_encoder
    model_path = st.session_state.model_path
    get_season = st.session_state.get_season # Access the helper function
    # Access the cached model from session state
    xgboost_model = st.session_state.xgboost_model
except AttributeError:
    st.error("Application state (data, encoders, model) not initialized. Please run from the main app.py.")
    st.stop() # Stop execution if essential state is missing

st.header("🔮 Advanced Sales Forecasting")
st.markdown("Predict future sales using a powerful XGBoost model and understand its predictions.")

# Prepare data for ML model (using encoded columns)
ml_df = df.copy()

# --- Define Features (CRITICAL: ensure these match what the model expects) ---
features = [
    'Day', 'Month', 'Year', 'Store', 'Dept', 'Type_Encoded', 'Size',
    'IsHoliday', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2',
    'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment', 'Season_Encoded'
]
X = ml_df[features]
y = ml_df['Weekly_Sales']

# --- Model Loading / Availability Check ---
# Assuming the main app.py handles the primary loading/training of xgboost_model
# and stores it in st.session_state.
# This section ensures the model is valid and its features match.
if xgboost_model is None:
    st.error("XGBoost model is not available in session state. Please ensure it's loaded/trained by the main application.")
    st.stop() # Stop if no model

# IMPORTANT: Check if the loaded model's feature names match current 'features' list.
# If they don't, it implies a mismatch between the features used for training and what's expected now.
if hasattr(xgboost_model, 'feature_names_in_') and list(xgboost_model.feature_names_in_) != features:
    st.warning("Loaded model's feature names do not match the current feature set. This might lead to incorrect predictions. Please re-run the main app to ensure model consistency.")
    # You might consider forcing a retraining here or disabling forecasting until resolved.
    # For now, we'll proceed but issue a warning.
    # If the app.py always trains on X_full, then X_train, y_train split here is only for evaluation display.

# Make predictions on the entire dataset for visualization
all_preds = xgboost_model.predict(X)
ml_df['XGBoost_Pred'] = all_preds

# --- Model Evaluation & Visualization ---
st.markdown("### Model Performance (XGBoost)")

# Use X_train and y_train for evaluation metrics if the model was trained on X_train
# If the model in session state was trained on the full dataset (X,y), use X,y here.
# For consistency with the previous script's model training on X_full,
# we'll use X and y for the evaluation metrics in this context.
# However, for a proper train/test split evaluation, it's better to explicitly use X_test, y_test.
# Let's keep the split for explicit evaluation metrics on unseen data.
X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(X, y, test_size=0.2, random_state=42)
y_test_pred = xgboost_model.predict(X_test_eval)

col_m1, col_m2 = st.columns(2)
with col_m1:
    fig_eval = go.Figure()
    # Plotting historical data from the main dataframe
    fig_eval.add_trace(go.Scatter(x=ml_df['Date'], y=ml_df['Weekly_Sales'], mode='lines', name='Actual Sales'))
    fig_eval.add_trace(go.Scatter(x=ml_df['Date'], y=ml_df['XGBoost_Pred'], mode='lines', name='Model Fit on Historical Data'))
    fig_eval.update_layout(title="Weekly Sales: Actual vs. Model Fit",
                            xaxis_title='Date', yaxis_title='Sales')
    st.plotly_chart(fig_eval, use_container_width=True)

with col_m2:
    st.markdown("##### Evaluation Metrics (on Test Set)")
    r2 = r2_score(y_test_eval, y_test_pred)
    mae = mean_absolute_error(y_test_eval, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test_eval, y_test_pred))
    metrics_data = {
        "Metric": ["R-squared (R²)", "Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)"],
        "Value": [f"{r2:.4f}", f"{mae:,.2f}", f"{rmse:,.2f}"]
    }
    st.table(pd.DataFrame(metrics_data))

    st.markdown("##### Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': xgboost_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    fig_importance = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                             title='XGBoost Feature Importance',
                             color='Importance', color_continuous_scale=px.colors.sequential.Viridis)
    fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_importance, use_container_width=True)

# --- SHAP Explanations (XGBoost Specific) ---
st.subheader("🔍 SHAP Feature Explanations (XGBoost)")
st.markdown("Understand how each feature contributes to the sales predictions. SHAP plots can be computationally intensive.")
if st.checkbox("Show SHAP Summary Plot (may take time for large datasets)"):
    # Using a smaller sample of X_train for SHAP calculation to improve performance
    # Adjust sample size based on your dataset size and desired responsiveness.
    shap_sample_size = min(5000, len(X_train_eval)) # Sample up to 5000 rows for SHAP
    X_shap_sample = X_train_eval.sample(n=shap_sample_size, random_state=42)

    st.info(f"Generating SHAP plots for a sample of {shap_sample_size} data points... This might take a moment.")
    try:
        explainer = shap.TreeExplainer(xgboost_model)
        # Check if the model has a 'feature_names_in_' attribute for explainer if needed,
        # or ensure X_shap_sample has column names consistent with training.
        shap_values = explainer.shap_values(X_shap_sample)

        # Matplotlib plots are rendered better with a figure object
        fig_shap_bar, ax_shap_bar = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_values, X_shap_sample, plot_type="bar", show=False, ax=ax_shap_bar)
        ax_shap_bar.set_title("SHAP Feature Importance (Overall Impact)")
        st.pyplot(fig_shap_bar, bbox_inches='tight')
        plt.close(fig_shap_bar) # Close the figure to free up memory

        fig_beeswarm, ax_beeswarm = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_values, X_shap_sample, show=False, ax=ax_beeswarm)
        ax_beeswarm.set_title("SHAP Feature Contributions (Beeswarm Plot)")
        st.pyplot(fig_beeswarm, bbox_inches='tight')
        plt.close(fig_beeswarm) # Close the figure to free up memory

    except Exception as e:
        st.error(f"Error generating SHAP plots: {e}. Please ensure SHAP is compatible with your XGBoost version and data, and try reducing the sample size if it's a memory issue.")
        st.exception(e) # Display full traceback for debugging

# --- Future Sales Forecast with user-selected Store/Dept ---
st.markdown("---")
st.markdown("### 📆 Future Sales Forecast for Specific Store/Department")

# --- Forecast Settings in Sidebar (Specific to this page) ---
with st.sidebar:
    st.markdown("---")
    st.header("🔮 Forecast Settings")
    st.markdown("Configure the future sales forecast.")
    # Use unique keys for selectboxes and slider to avoid conflicts
    forecast_store = st.selectbox("Forecast for Store", sorted(df['Store'].unique()), index=0, key='forecast_store_sidebar_forecast_page')
    forecast_dept = st.selectbox("Forecast for Department", sorted(df['Dept'].unique()), index=0, key='forecast_dept_sidebar_forecast_page')
    future_days_specific = st.slider("Forecast for how many days?", 7, 365, 90, key='future_days_sidebar_forecast_page') # Renamed to avoid clash

st.markdown(f"Forecasting sales for **Store: {forecast_store}** and **Department: {forecast_dept}** for the next **{future_days_specific}** days.")

# Get a representative row for the selected store/department from ml_df (which has encoded columns)
representative_row = ml_df[(ml_df['Store'] == forecast_store) & (ml_df['Dept'] == forecast_dept)].tail(1)

if representative_row.empty:
    st.warning(f"No historical data found for Store {forecast_store} and Department {forecast_dept}. Please select a different combination or upload more data.")
else:
    last_known_data = representative_row.iloc[0].to_dict()

    future_dates_forecast = pd.date_range(ml_df['Date'].max() + timedelta(days=1), periods=future_days_specific)

    # --- OPTIMIZATION: Vectorize future data generation for specific forecast ---
    # Create base DataFrame for future dates
    future_date_data = pd.DataFrame({
        'Date': future_dates_forecast,
        'Day': [d.day for d in future_dates_forecast],
        'Month': [d.month for d in future_dates_forecast],
        'Year': [d.year for d in future_dates_forecast]
    })
    future_date_data['Season'] = future_date_data['Month'].map(get_season)
    # Ensure season_encoder can transform future seasons
    # (assuming season_encoder is already globally fitted on all known seasons from main app)
    # If a new season arises that wasn't in original data, this might fail unless encoder is updated
    future_date_data['Season_Encoded'] = season_encoder.transform(future_date_data['Season'])

    # Replicate the specific store/dept/type/size info for all future dates
    # Only pick relevant features from last_known_data that are constant for this specific forecast
    constant_features = ['Store', 'Dept', 'Type_Encoded', 'Size']
    for feature in constant_features:
        future_date_data[feature] = last_known_data[feature]

    # Add latest known external features (Temperature, Fuel_Price, CPI, Unemployment)
    # These are taken from the LAST known row for the SELECTED store/dept.
    # If external factors should be overall latest, use df.tail(1)[...].iloc[0].to_dict() instead.
    external_features = ['Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment']
    for feature in external_features:
        # Default MarkDowns to 0.0 for future unless explicit future data is available
        if feature.startswith('MarkDown'):
            future_date_data[feature] = 0.0
        else:
            future_date_data[feature] = last_known_data.get(feature, 0.0) # Use .get with default 0.0 for safety

    future_date_data['IsHoliday'] = 0 # Default non-holiday

    future_X_df = future_date_data # This is now the dataframe ready for prediction
    # Ensure column order and types match training data for prediction
    future_X_df_for_pred = future_X_df[features]

    # Predict future sales
    future_preds = xgboost_model.predict(future_X_df_for_pred)
    future_X_df['Weekly_Sales'] = future_preds

    # --- Plotting Specific Forecast ---
    fig_future = go.Figure()
    # Plot historical sales for the *selected* store/department
    historical_sales_selected_sd = ml_df[(ml_df['Store'] == forecast_store) & (ml_df['Dept'] == forecast_dept)]
    fig_future.add_trace(go.Scatter(x=historical_sales_selected_sd['Date'], y=historical_sales_selected_sd['Weekly_Sales'],
                                    name='Historical Sales', mode='lines+markers', line=dict(color='blue')))

    # To make the future forecast line continuous with historical, add the last historical point
    last_hist_point = historical_sales_selected_sd.tail(1)[['Date', 'Weekly_Sales']].rename(columns={'Weekly_Sales': 'Weekly_Sales_Predicted'})
    forecast_with_tail = pd.concat([last_hist_point, future_X_df[['Date', 'Weekly_Sales']].rename(columns={'Weekly_Sales': 'Weekly_Sales_Predicted'})], ignore_index=True)
    forecast_with_tail = forecast_with_tail.drop_duplicates(subset=['Date'], keep='first') # Keep first to prefer historical

    fig_future.add_trace(go.Scatter(x=forecast_with_tail['Date'], y=forecast_with_tail['Weekly_Sales_Predicted'],
                                     name='Future Forecast', mode='lines', line=dict(color='red', dash='dash')))

    fig_future.update_layout(title=f"Future Sales Forecast for Store {forecast_store}, Dept {forecast_dept}",
                              xaxis_title='Date', yaxis_title='Forecasted Sales')
    st.plotly_chart(fig_future, use_container_width=True)

    # --- Forecast Details Table ---
    st.markdown("##### Detailed Future Forecast")
    st.dataframe(future_X_df[['Date', 'Weekly_Sales']].set_index('Date').style.format({"Weekly_Sales": "${:,.2f}"}))

    csv_forecast = future_X_df[['Date', 'Weekly_Sales']].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Future Forecast as CSV",
        data=csv_forecast,
        file_name=f'sales_forecast_store{forecast_store}_dept{forecast_dept}_{future_days_specific}days.csv',
        mime='text/csv',
    )

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ by AlienMinus | Powered by Streamlit & Plotly")