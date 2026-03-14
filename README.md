# Temperature Forecasting using Econometric Models

This project develops a temperature forecasting system using econometric and time-series modelling techniques. The objective is to predict maximum and minimum air temperatures using historical meteorological data and statistical forecasting methods.

The project combines econometric modelling with an interactive dashboard to allow users to generate temperature predictions based on weather conditions.

---

## Project Overview

Temperature forecasting plays an important role in areas such as:

* Agriculture planning
* Energy management
* Infrastructure planning
* Climate monitoring

This project applies econometric techniques commonly used in financial modelling to weather data in order to capture temporal dependencies, seasonal patterns, and volatility dynamics.

---

## Models Implemented

The forecasting system integrates three statistical models.

### Ordinary Least Squares (OLS)

OLS regression is used to model the relationship between temperature and meteorological variables such as:

* Sunshine duration
* Soil temperature
* Rainfall
* Grass minimum temperature
* Wind speed

Lagged temperature variables and seasonal transformations are included to capture temporal dynamics.

---

### ARIMA Model

ARIMA models are used to capture time-series dependencies in temperature data.

Model specification used:

ARIMA (1,0,1)

These models help capture autoregressive behaviour and short-term fluctuations in temperature.

---

### GARCH Model

GARCH models are used to estimate **time-varying volatility** in temperature forecasts.

This helps quantify uncertainty in predictions and identify periods where temperature forecasts may be less stable.

---

## Hybrid Forecasting Model

The final temperature prediction combines OLS and ARIMA forecasts using a weighted ensemble approach:

Hybrid Forecast = 0.7 × OLS Forecast + 0.3 × ARIMA Forecast

This approach improves forecast stability by combining structural modelling and time-series dynamics.

---

## Dataset

The dataset consists of approximately **50 years of meteorological observations** containing more than **18,000 records**.

Variables include:

* Sunshine hours
* Soil temperature
* Precipitation
* Grass minimum temperature
* Wind speed
* Maximum temperature
* Minimum temperature

These variables are used to construct lag features and seasonal indicators for forecasting models.

---

## Interactive Dashboard

An interactive **Streamlit dashboard** was developed to allow users to generate temperature forecasts.

Dashboard features include:

* User input interface for weather conditions
* Temperature prediction using econometric models
* Model diagnostics and evaluation metrics
* Data visualisations including correlation heatmaps and temperature trends

---

## Technologies Used

Python
Pandas
NumPy
Statsmodels
Matplotlib
Seaborn
Streamlit

---
## My Contribution

This project was completed as part of a group assignment for the **Financial Econometrics (FIN41660)** module.

My primary contributions to the project included:

1) Developing the Python modelling pipeline for temperature forecasting
2) Implementing econometric models including **OLS, ARIMA, and GARCH**
3) Performing data preprocessing and feature engineering (lag variables, seasonal transformations)
4) Conducting model evaluation using metrics such as **R², MAE, and RMSE**
5) Contributing to the development of the **interactive Streamlit dashboard** used for generating temperature forecasts
6) Creating visualisations and model diagnostic plots

The project involved collaboration with other team members who contributed to the dashboard interface and report writing.
---

## Project Report

The full econometrics project report is available below:

[View the report](Financial Econometrics- FIN41660.pdf)

Link to our Dashboard
https://temperature-dashboard-4tdkplpgumshtywmjgdnlz.streamlit.app/
---

## Author

Shruti Avinash Punjabi
