import os
import requests
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
session = os.getenv("MYFXBOOK_SESSION")
id = os.getenv("MYFXBOOK_ID")

# Streamlit configuration
st.set_page_config(layout="wide")  # Enable wide mode

# Fetch Data
url = f"https://www.myfxbook.com/api/get-data-daily.json?session={session}&id={id}&start=2000-01-01&end=2025-11-01"
response = requests.get(url)
data = response.json()

# Check if dataDaily is available
if data.get("dataDaily") and len(data["dataDaily"]) > 0:
    daily_data = data["dataDaily"]
    date_balance = [(entry[0]["date"], entry[0]["balance"]) for entry in daily_data]
    df = pd.DataFrame(date_balance, columns=["Date", "Balance"])

    # Convert Date to datetime format
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df.set_index("Date", inplace=True)

    # Prepare and train model
    df["Days"] = (df.index - df.index[0]).days
    X = df[["Days"]]
    y = df["Balance"]
    model = LinearRegression()
    model.fit(X, y)

    # Forecasting
    future_days = np.arange(df["Days"].max() + 1, df["Days"].max() + 31).reshape(-1, 1)
    optimistic_forecast = model.predict(future_days) * 1.05
    conservative_forecast = model.predict(future_days) * 0.95
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=30, freq="D")
    
    # Append projections to actual data for continuity
    extended_dates = df.index.append(future_dates)
    combined_optimistic = np.concatenate([df["Balance"].values, optimistic_forecast.flatten()])
    combined_conservative = np.concatenate([df["Balance"].values, conservative_forecast.flatten()])

    # Plotting with Plotly
    fig = go.Figure()

    # Actual Balance
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Balance"], mode='lines', name="Actual Balance", line=dict(color="blue")
    ))

    # Optimistic Projection
    fig.add_trace(go.Scatter(
        x=extended_dates, y=combined_optimistic, mode='lines', name="Optimistic Projection", line=dict(color="green", dash="dash")
    ))

    # Conservative Projection
    fig.add_trace(go.Scatter(
        x=extended_dates, y=combined_conservative, mode='lines', name="Conservative Projection", line=dict(color="red", dash="dash")
    ))

    # Customize layout
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Balance",
        height=700  # Adjust the height to make it more screen-filling in wide mode
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("No data available to display. Please check the API response or parameters.")
