# Time-Series-Sales-Forecasting-Python
End-to-end retail sales time-series forecasting and visualization using Python, SARIMA, Prophet, and Streamlit.
# Big Data Time-Series Forecasting & Visualization

## Overview
This project implements an end-to-end time-series forecasting and visualization system for retail sales data.  
It combines statistical forecasting models with interactive visual analytics to support business decision-making.

## Key Features
 Complete forecasting pipeline: data preprocessing, modeling, evaluation, and visualization
 Comparison of ARIMA, SARIMA, Prophet, and Seasonal Naïve models
 Prediction intervals for uncertainty-aware forecasting
 Interactive Streamlit dashboard for business users

## Dataset
 Retail transactional sales time-series
 4.5 years of historical data
 Aggregated at daily, weekly, and monthly levels

## Models Implemented
 Seasonal Naïve (baseline)
 ARIMA
 SARIMA (best-performing model)
 Prophet

## Results
 SARIMA achieved the lowest RMSE, MAE, and AIC
 Strong handling of weekly seasonality
 Residual diagnostics confirm statistically adequate model fit

## Streamlit Dashboard
The Streamlit app allows users to:
 Visualize historical sales and forecasts
 Explore seasonality and trends
 View prediction intervals and residual diagnostics

## Technologies
Python, Pandas, NumPy, Statsmodels, Prophet, Matplotlib, Plotly, Streamlit

## Thesis
The full MSc thesis PDF is available in the `docs/` folder.
