# temp_predictor.py
"""
TemperaturePredictor module
Provides a simple pipeline: load data, create features, train OLS/ARIMA/GARCH,
compute forecast metrics, and single-date forecasts for use in a Streamlit app.
"""

import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class TemperaturePredictor:
    REQUIRED_COLS = ['date', 'sun', 'soil', 'rain', 'gmin', 'wdsp', 'maxtp', 'mintp']

    def __init__(self, file_path: str = "Weather_final_cleaned.csv"):
        self.file_path = file_path
        self.df = None
        self.train = None
        self.test = None
        self.ols_models = {}
        self.arima_models = {}
        self.garch_models = {}
        self.predictions = {}

    def load_data(self):
        df = pd.read_csv(self.file_path)
        # basic checks
        missing_cols = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['date']).set_index('date').sort_index()
        # ensure numeric columns
        for c in ['sun','soil','rain','gmin','wdsp','maxtp','mintp']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['sun','soil','rain','gmin','wdsp','maxtp','mintp'])
        self.df = df
        return df

    def create_features(self):
        if self.df is None:
            raise RuntimeError("Load data first with load_data()")
        df = self.df.copy()
        # lag features
        cols = ['sun','soil','rain','gmin','wdsp','maxtp','mintp']
        for c in cols:
            df[f"{c}_lag1"] = df[c].shift(1)
        for c in ['maxtp','mintp']:
            df[f"{c}_lag2"] = df[c].shift(2)
            df[f"{c}_lag7"] = df[c].shift(7)
        # seasonal features
        df['month'] = df.index.month
        df['dayofyear'] = df.index.dayofyear
        df['sin_year'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
        df['cos_year'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
        df = df.dropna()
        self.df = df
        return df

    def split_data(self, ratio=0.8):
        if self.df is None:
            raise RuntimeError("Create features first with create_features()")
        split_idx = int(len(self.df) * ratio)
        self.train = self.df.iloc[:split_idx]
        self.test = self.df.iloc[split_idx:]
        return self.train, self.test

    def train_ols(self, target='maxtp'):
        predictors = [
            "sun_lag1", "soil_lag1", "rain_lag1", "gmin_lag1", "wdsp_lag1",
            f"{target}_lag1", f"{target}_lag2", f"{target}_lag7", "sin_year", "cos_year"
        ]
        X_train = add_constant(self.train[predictors]).rename(columns=str)
        y_train = self.train[target]
        X_test = add_constant(self.test[predictors]).rename(columns=str)
        y_test = self.test[target]
        model = OLS(y_train, X_train).fit()
        self.ols_models[target] = model
        pred_test = model.predict(X_test)
        self.predictions[target] = {'actual': y_test, 'predicted': pred_test}
        return model

    def train_arima(self, target='maxtp', order=(1,0,1)):
        series = self.train[target]
        try:
            model = ARIMA(series, order=order).fit()
            self.arima_models[target] = model
            return model
        except Exception as e:
            self.arima_models[target] = None
            return None

    def arima_forecast_one(self, target='maxtp'):
        model = self.arima_models.get(target)
        if model is None:
            return None
        try:
            return float(model.forecast(1)[0])
        except Exception:
            return None

    def train_garch(self, target='maxtp'):
        model_ols = self.ols_models.get(target)
        if model_ols is None:
            return None
        # residuals from OLS on training set
        X = model_ols.model.exog
        resid = self.train[target] - model_ols.predict(X)
        try:
            am = arch_model(resid, vol='Garch', p=1, q=1).fit(disp='off')
            self.garch_models[target] = am
            return am
        except Exception:
            self.garch_models[target] = None
            return None

    def run_pipeline(self, ratio=0.8, arima_order=(1,0,1)):
        self.load_data()
        self.create_features()
        self.split_data(ratio)
        # train for both targets
        for t in ['maxtp','mintp']:
            self.train_ols(t)
            self.train_arima(t, order=arima_order)
            self.train_garch(t)
        return self

    def get_metrics_df(self):
        rows = []
        for t in ['maxtp','mintp']:
            actual = self.predictions[t]['actual']
            pred = self.predictions[t]['predicted']
            r2 = r2_score(actual, pred)
            mae = mean_absolute_error(actual, pred)
            rmse = np.sqrt(mean_squared_error(actual, pred))
            mask = actual != 0
            mape = np.nan
            if mask.sum() > 0:
                mape = (np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)
            rows.append({'Target': t.upper(), 'R2': r2, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape})
        return pd.DataFrame(rows)

    def monthly_average(self):
        return self.df[['maxtp','mintp']].groupby(self.df.index.month).mean()

    def predict_from_dict(self, params: dict):
        """
        params should contain keys:
          sun, soil, rain, gmin, wdsp, maxtp_yesterday, mintp_yesterday, forecast_date (datetime)
        returns dict with OLS/ARIMA/Hybrid forecasts for maxtp and mintp.
        """
        if self.ols_models == {}:
            raise RuntimeError("Train models first (run_pipeline or train_ols).")
        fd = params.get('forecast_date', pd.Timestamp.today())
        if not isinstance(fd, pd.Timestamp):
            fd = pd.to_datetime(fd)
        doy = fd.dayofyear
        sin_year = np.sin(2*np.pi*doy/365.25)
        cos_year = np.cos(2*np.pi*doy/365.25)

        base = {
            'const': 1.0,
            'sun_lag1': params.get('sun', 0.0),
            'soil_lag1': params.get('soil', 0.0),
            'rain_lag1': params.get('rain', 0.0),
            'gmin_lag1': params.get('gmin', 0.0),
            'wdsp_lag1': params.get('wdsp', 0.0),
            'maxtp_lag1': params.get('maxtp_yesterday', 0.0),
            'maxtp_lag2': params.get('maxtp_yesterday', 0.0),
            'maxtp_lag7': params.get('maxtp_yesterday', 0.0),
            'mintp_lag1': params.get('mintp_yesterday', 0.0),
            'mintp_lag2': params.get('mintp_yesterday', 0.0),
            'mintp_lag7': params.get('mintp_yesterday', 0.0),
            'sin_year': sin_year,
            'cos_year': cos_year,
            'month': fd.month,
            'dayofyear': doy
        }

        out = {}
        for t in ['maxtp','mintp']:
            model = self.ols_models[t]
            cols = [str(c) for c in model.model.exog_names]
            row = {c: float(base.get(c, 0.0)) for c in cols}
            X = pd.DataFrame([row], columns=cols)
            ols_val = float(model.predict(X)[0])
            arima_val = self.arima_forecast_one(t)
            hybrid = 0.7 * ols_val + 0.3 * (arima_val if arima_val is not None else ols_val)
            out[t] = {'OLS': ols_val, 'ARIMA': arima_val, 'Hybrid': hybrid}
        return out
