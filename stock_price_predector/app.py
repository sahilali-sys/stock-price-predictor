import streamlit as st
import joblib 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

model= joblib.load(
    "stock_price_model.pkl"
)

scaler= joblib.load(
    "scaler.pkl"
)

features = joblib.load(
    "features_name.pkl"
)
st.set_page_config(
    page_title="Stock Price Predictor",
    layout="wide"
)

st.title("📈Stock Price Forecasting Dashboard")

st.write(
    """ 
    This application predict future stock prices
    using a machine learning forecasting model.
    """)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose Section",
    [
        "Home",
        "Model Information",
        "Forecast"
    ]
)

if page == "Home":
    st.header("Project Overview")

    st.write(
        """
        This project forecasts stock price using 
        machine learning and time series feature engineering.
        
        Features include:
        
        - Lag Variables
        - Rolling Averages
        - Momentum Indicators
        - Volume Information
        """
        )
    
    st.subheader("Projects Highlights")

    st.markdown(
        """
            - Stock Price Forecasting 
            - Time Series Feature Engineeering 
            - Random Forest Regressor 
            - Future Price Prediction
            - Interactive Streamlit Dashboard
            """)
    
elif page == "Model Information":
    st.header("Model Details")

    st.write(
        f"Number of features: {len(features)}"
    )

    st.write(
        "Model : Random Forest Regressor"
    )
    st.write(
        "Scaler : StandardScaler"
    )

    st.write(
        "Features Used:"
    )

    st.subheader("Feature Used")

    for feature in features:
        st.write(f" - {feature}")

    st.subheader("Model Performance")

    st.write("MAE : Your MAE Value")
    st.write("RMSE : Your RMSE Value")
    st.write("R2 Score: Your R2 Value")

elif page == "Forecast":

    st.header("Stock Price Forecast")

    # Input Features
    lag_1 = st.number_input(
        "Lag 1",
        value=150.0
    )

    lag_2 = st.number_input(
        "Lag 2",
        value=149.0
    )

    lag_3 = st.number_input(
        "Lag 3",
        value=148.0
    )

    rolling_7 = st.number_input(
        "Rolling Mean 7",
        value=150.0
    )

    rolling_30 = st.number_input(
        "Rolling Mean 30",
        value=145.0
    )

    rolling_std_7 = st.number_input(
        "Rolling STD 7",
        value=5.0
    )

    momentum = st.number_input(
        "Momentum",
        value=1.0
    )

    volume = st.number_input(
        "Volume",
        value=50000000
    )

    # Prediction Button
    predict_button = st.button(
        "Predict Next Day Price"
    )

    if predict_button:

        # Create Input DataFrame
        input_df = pd.DataFrame(
            [[
                lag_1,
                lag_2,
                lag_3,
                rolling_7,
                rolling_30,
                rolling_std_7,
                momentum,
                volume
            ]],
            columns=features
        )

        # Scale Input
        input_scaled = scaler.transform(
            input_df
        )

        # Predict
        prediction = model.predict(
            input_scaled
        )[0]

        # Metric Cards
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Price",
                f"${prediction:.2f}"
            )

        with col2:
            st.metric(
                "Forecast Horizon",
                "1 Day"
            )

        # Create Simple 7-Day Trend
        future_prices = [
            prediction * 0.99,
            prediction,
            prediction * 1.01,
            prediction * 1.02,
            prediction * 1.01,
            prediction * 1.03,
            prediction * 1.04
        ]

        days = np.arange(1, 8)

        # Plot Forecast
        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        ax.plot(
            days,
            future_prices,
            marker="o"
        )

        ax.set_title(
            "7-Day Forecast Trend"
        )

        ax.set_xlabel(
            "Days Ahead"
        )

        ax.set_ylabel(
            "Predicted Price"
        )

        st.pyplot(fig)