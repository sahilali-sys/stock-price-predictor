# 📈 Stock Price Predictor

An end-to-end Machine Learning project that predicts future stock prices using time-series feature engineering and a Random Forest Regression model. The project also includes an interactive Streamlit dashboard for making predictions and visualizing forecast trends.

---

## 🚀 Project Overview

Stock prices are influenced by historical trends and market behavior. This project uses previous stock data to predict the next day's closing price.

The workflow includes:

- Data Collection
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training
- Model Evaluation
- Streamlit Dashboard Development

---

## 📊 Features

### Time-Series Features

The model uses:

- Lag_1 (Previous Day Price)
- Lag_2
- Lag_3
- Rolling_Mean_7
- Rolling_Mean_30
- Rolling_STD_7
- Momentum
- Volume

### Machine Learning

- Random Forest Regressor
- StandardScaler
- Train/Test Split
- Model Persistence using Joblib

### Dashboard

- Interactive Inputs
- Next-Day Price Prediction
- Forecast Metrics
- 7-Day Forecast Visualization
- Model Information Page

---

## 🛠️ Tech Stack

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- Joblib
- Streamlit

---

## 📂 Project Structure

```text
stock-price-predictor/
│
├── app.py
├── stock_price_model.pkl
├── scaler.pkl
├── features_name.pkl
├── requirements.txt
├── README.md
└── screenshots/
```
