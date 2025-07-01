# pages/1_Dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Access global data and encoders from st.session_state
df = st.session_state.df
type_encoder = st.session_state.type_encoder
season_encoder = st.session_state.season_encoder

st.header("📊 Sales Performance Dashboard")
st.markdown("Explore key sales metrics, trends, and distributions across your retail data.")

# --- Sidebar Filters ---
with st.sidebar:
    st.header("⚙️ Data Filters")
    st.markdown("Adjust the parameters below to refine your data view.")

    # Use original column values for display in multiselects
    unique_stores = sorted(df['Store'].unique())
    unique_depts = sorted(df['Dept'].unique())
    unique_types = sorted(df['Type'].unique())
    unique_seasons = sorted(df['Season'].unique())
    unique_months = sorted(df['Date'].dt.month.unique())

    selected_stores = st.multiselect("Select Store(s)", unique_stores, default=unique_stores)
    selected_depts = st.multiselect("Select Department(s)", unique_depts, default=unique_depts)
    selected_types = st.multiselect("Select Store Type(s)", unique_types, default=unique_types)
    selected_seasons = st.multiselect("Select Season(s)", unique_seasons, default=unique_seasons)
    selected_months = st.multiselect("Select Month(s)", unique_months, default=unique_months)
    selected_is_holiday = st.radio("Holiday Impact", ['All', 'Holiday Only', 'Non-Holiday'], index=0)

# --- Data Filtering ---
filtered_df = df[
    df['Store'].isin(selected_stores) &
    df['Dept'].isin(selected_depts) &
    df['Type'].isin(selected_types) &
    df['Season'].isin(selected_seasons) &
    df['Date'].dt.month.isin(selected_months)
]
if selected_is_holiday == 'Holiday Only':
    filtered_df = filtered_df[filtered_df['IsHoliday'] == True]
elif selected_is_holiday == 'Non-Holiday':
    filtered_df = filtered_df[filtered_df['IsHoliday'] == False]

if filtered_df.empty:
    st.warning("No data found for the selected filters. Please adjust your selections.")
    st.stop()

# --- KPI Metrics ---
st.markdown("### 📌 Key Performance Indicators")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Sales (Filtered)", f"${filtered_df['Weekly_Sales'].sum():,.2f}")
kpi2.metric("Avg. Weekly Sales (Filtered)", f"${filtered_df['Weekly_Sales'].mean():,.2f}")
kpi3.metric("Unique Stores (Filtered)", f"{filtered_df['Store'].nunique()}")
kpi4.metric("Unique Departments (Filtered)", f"{filtered_df['Dept'].nunique()}")

# --- Sales Trends & Distributions ---
st.markdown("---")
st.markdown("## 📈 Sales Performance Analysis")
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Weekly Sales Over Time")
    time_series_data = filtered_df.groupby('Date')['Weekly_Sales'].sum().reset_index()
    fig1 = px.line(time_series_data, x='Date', y='Weekly_Sales', title='Total Weekly Sales Trend', markers=True)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown("#### Sales by Store")
    store_sales = filtered_df.groupby('Store')['Weekly_Sales'].sum().reset_index().sort_values(by='Weekly_Sales', ascending=False)
    fig2 = px.bar(store_sales, x='Store', y='Weekly_Sales', title='Total Sales per Store',
                     color='Weekly_Sales', color_continuous_scale=px.colors.sequential.Plasma)
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    st.markdown("#### Sales Distribution by Store Type")
    fig3 = px.box(filtered_df, x='Type', y='Weekly_Sales', color='Type',
                     title='Weekly Sales Distribution by Store Type')
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.markdown("#### Holiday vs. Non-Holiday Sales")
    holiday_sales_summary = filtered_df.groupby('IsHoliday')['Weekly_Sales'].sum().reset_index()
    holiday_sales_summary['IsHoliday_Label'] = holiday_sales_summary['IsHoliday'].map({True: 'Holiday', False: 'Non-Holiday'})
    fig4 = px.pie(holiday_sales_summary, names='IsHoliday_Label', values='Weekly_Sales',
                     title='Sales Share: Holiday vs Non-Holiday Weeks', hole=0.3)
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("#### Season-wise Weekly Sales")
season_sales_summary = filtered_df.groupby('Season')['Weekly_Sales'].sum().reset_index().sort_values(by='Weekly_Sales', ascending=False)
fig6 = px.bar(season_sales_summary, x='Season', y='Weekly_Sales', color='Season',
              title='Total Sales by Season', text_auto=True)
st.plotly_chart(fig6, use_container_width=True)

st.markdown("#### Monthly Sales Trends by Store Type")
filtered_df_for_plot = filtered_df.copy()
filtered_df_for_plot.loc[:, 'MonthName'] = filtered_df_for_plot['Date'].dt.strftime('%b')
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_sales_trend = filtered_df_for_plot.groupby(['MonthName', 'Type'])['Weekly_Sales'].sum().reset_index()
monthly_sales_trend['MonthName'] = pd.Categorical(monthly_sales_trend['MonthName'], categories=month_order, ordered=True)
fig7 = px.line(monthly_sales_trend.sort_values('MonthName'), x='MonthName', y='Weekly_Sales', color='Type',
               title='Monthly Sales Trends by Store Type', markers=True)
st.plotly_chart(fig7, use_container_width=True)

# --- Correlation Heatmap ---
st.markdown("---")
st.markdown("## 📊 Data Relationships")
st.markdown("Understand the correlation between numerical features in your dataset.")
fig5, ax = plt.subplots(figsize=(12, 8))
corr_df = df.select_dtypes(include=np.number).drop(columns=['Day', 'Month', 'Year'], errors='ignore').corr()
sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, ax=ax)
ax.set_title('Correlation Heatmap of Numerical Features')
st.pyplot(fig5)

# --- Aggregated Summary Table ---
st.markdown("---")
st.markdown("## 📋 Detailed Data Summary")
st.markdown("Aggregated view of sales by selected dimensions.")
aggr_table = filtered_df.groupby(['Store', 'Dept', 'Type', 'Season']).agg(
    Total_Sales=('Weekly_Sales', 'sum'),
    Average_Sales=('Weekly_Sales', 'mean'),
    Max_Sales=('Weekly_Sales', 'max'),
    Min_Sales=('Weekly_Sales', 'min')
).reset_index()
st.dataframe(aggr_table.style.format({"Total_Sales": "${:,.2f}", "Average_Sales": "${:,.2f}", "Max_Sales": "${:,.2f}", "Min_Sales": "${:,.2f}"}))

# --- View Raw Data ---
with st.expander("📄 View Filtered Raw Data"):
    st.dataframe(filtered_df)
    csv_filtered = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv_filtered,
        file_name='filtered_sales_data.csv',
        mime='text/csv',
    )

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ by AlienMinus | Powered by Streamlit & Plotly")