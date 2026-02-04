import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(page_title="Apple Stock Price Prediction", layout="wide")

st.title("Apple Stock Price Forecasting")
st.write("SARIMA-based Time Series Forecasting")


# Load data (FIXED CSV)

@st.cache_data
def load_data():
    df = pd.read_csv("P625 DATASET.csv")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Convert Date column
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort & set index
    df = df.sort_values("Date")
    df.set_index("Date", inplace=True)

    # Set business-day frequency
    df = df.asfreq("B")

    # Forward fill missing values
    df["Close"] = df["Close"].ffill()

    return df

df = load_data()

# Historical Visualization

st.subheader("Historical Close Prices")
st.line_chart(df["Close"])


# Train-Test Split (70%-30%)

train_size = int(len(df) * 0.7)

train = df["Close"].iloc[:train_size]
test = df["Close"].iloc[train_size:]


# SARIMA Model
st.subheader("SARIMA Model Training")

sarima_model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 5)
)

sarima_fit = sarima_model.fit()


# Prediction on Test Data

test_pred = sarima_fit.forecast(steps=len(test))

# Model Evaluation

rmse = np.sqrt(np.mean((test - test_pred) ** 2))
mae = np.mean(np.abs(test - test_pred))

st.subheader("Model Evaluation Metrics")
st.write(f"**RMSE:** {rmse:.2f}")
st.write(f"**MAE:** {mae:.2f}")


# Test vs Prediction Plot

st.subheader("Actual vs Predicted Prices")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(train.index, train, label="Train")
ax.plot(test.index, test, label="Test", color="orange")
ax.plot(test.index, test_pred, label="Predicted", color="green")
ax.legend()
ax.set_title("SARIMA Model Performance")

st.pyplot(fig)

# Future Forecast

st.subheader("Forecast Future Stock Prices")

days = st.slider("Select forecast days", 1, 60, 30)

future_forecast = sarima_fit.forecast(steps=days)

future_dates = pd.date_range(
    start=df.index[-1] + pd.Timedelta(days=1),
    periods=days,
    freq="B"
)

future_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Close Price": future_forecast.values
})

st.dataframe(future_df)

# Future Forecast Plot

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(df.index, df["Close"], label="Historical")
ax2.plot(future_dates, future_forecast, label="Forecast", marker="o")
ax2.legend()
ax2.set_title("Apple Stock Price – Next Days Forecast")

st.pyplot(fig2)
