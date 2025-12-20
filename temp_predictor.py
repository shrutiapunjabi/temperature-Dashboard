# %%
import warnings
warnings.filterwarnings("ignore")
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None

from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Dict, Any, Optional

# Set style for better-looking plots


# %%
class TemperaturePredictor:
    REQUIRED_COLS = ['date', 'sun', 'soil', 'rain', 'gmin', 'wdsp', 'maxtp', 'mintp']

    def __init__(self, file_path: str = "Data.csv"):
        self.file_path = file_path
        self.df = None
        self.train = None
        self.test = None

        self.ols_models = {}
        self.arima_models = {}
        self.garch_models = {}

        self.metrics = []
        self.predictions = {}

    # ----------------------------------------------------------
    # LOAD DATA
    # ----------------------------------------------------------
    def load_data(self):
        df = pd.read_csv(self.file_path)
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.set_index("date").sort_index()

        for c in ["sun", "soil", "rain", "gmin", "wdsp", "maxtp", "mintp"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["sun", "soil", "rain", "gmin", "wdsp", "maxtp", "mintp"])
        self.df = df

        print(f"✔ Loaded {len(df)} rows")

    # ----------------------------------------------------------
    # FEATURE ENGINEERING
    # ----------------------------------------------------------
    def create_features(self):
        df = self.df.copy()

        for col in ["sun", "soil", "rain", "gmin", "wdsp"]:
            df[f"{col}_lag1"] = df[col].shift(1)

        for col in ["maxtp", "mintp"]:
            df[f"{col}_lag1"] = df[col].shift(1)
            df[f"{col}_lag2"] = df[col].shift(2)
            df[f"{col}_lag7"] = df[col].shift(7)

        df["month"] = df.index.month
        df["dayofyear"] = df.index.dayofyear
        df["sin_year"] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
        df["cos_year"] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)

        df = df.dropna()
        self.df = df

        print(f"✔ Created features: {df.shape}")

    # ----------------------------------------------------------
    # SPLIT DATA
    # ----------------------------------------------------------
    def split_data(self, ratio=0.8):
        s = int(len(self.df) * ratio)
        self.train = self.df.iloc[:s]
        self.test = self.df.iloc[s:]
        print(f"Train: {self.train.shape}, Test: {self.test.shape}")

    # ----------------------------------------------------------
    # TRAIN OLS
    # ----------------------------------------------------------
    def train_ols(self, target):
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

        print(f"OLS ({target}) R² test = {r2_score(y_test, pred_test):.3f}")

    # ----------------------------------------------------------
    # TRAIN ARIMA
    # ----------------------------------------------------------
    def train_arima(self, target):
        series = self.train[target]
        try:
            model = ARIMA(series, order=(1,0,1)).fit()
            self.arima_models[target] = model
            print(f"✔ ARIMA trained for {target}")
        except:
            print(f"⚠ ARIMA failed for {target}")
            self.arima_models[target] = None
    def arima_forecast_one(self, target):
        try:
            series = self.df[target]  # full history, not just train
            model = ARIMA(series, order=(1,0,1)).fit()
            return float(model.forecast(1).iloc[0])
        except Exception as e:
            print(f"ARIMA forecast failed for {target}: {e}")
            return None


    # def arima_forecast_one(self, target):
    #     model = self.arima_models.get(target)
    #     if model is None:
    #         return None
    #     try:
    #         return float(model.forecast(1)[0])
    #     except:
    #         return None

    # ----------------------------------------------------------
    # TRAIN GARCH
    # ----------------------------------------------------------
    def train_garch(self, target):
        model = self.ols_models[target]
        X = pd.DataFrame(model.model.exog, index=self.train.index, columns=model.model.exog_names)
        resid = self.train[target] - model.predict(X)
        try:
            am = arch_model(resid, vol="Garch", p=1, q=1).fit(disp="off")
            self.garch_models[target] = am
            print(f"✔ GARCH trained for {target}")
        except:
            print(f"⚠ GARCH failed for {target}")

    # ----------------------------------------------------------
    # PREDICT METHOD (supports yesterday temps + forecast date)
    # ----------------------------------------------------------
    def predict_from_dict(self, params):
        forecast_date = params.get("forecast_date")
        if forecast_date:
            forecast_date = pd.to_datetime(forecast_date)
        else:
            forecast_date = pd.Timestamp.today()

        month = forecast_date.month
        dayofyear = forecast_date.dayofyear
        sin_year = np.sin(2*np.pi*dayofyear/365.25)
        cos_year = np.cos(2*np.pi*dayofyear/365.25)

        base = {
            "const": 1.0,
            "sun_lag1": params["sun"],
            "soil_lag1": params["soil"],
            "rain_lag1": params["rain"],
            "gmin_lag1": params["gmin"],
            "wdsp_lag1": params["wdsp"],
            "maxtp_lag1": params.get("maxtp_yesterday", 0),
            "maxtp_lag2": params.get("maxtp_yesterday", 0),
            "maxtp_lag7": params.get("maxtp_yesterday", 0),
            "mintp_lag1": params.get("mintp_yesterday", 0),
            "mintp_lag2": params.get("mintp_yesterday", 0),
            "mintp_lag7": params.get("mintp_yesterday", 0),
            "month": month,
            "dayofyear": dayofyear,
            "sin_year": sin_year,
            "cos_year": cos_year
        }

        out = {}

        for target in ["maxtp", "mintp"]:
            model = self.ols_models[target]
            cols = [str(c) for c in model.model.exog_names]

            row = {c: float(base.get(c, 0.0)) for c in cols}
            X = pd.DataFrame([row], columns=cols)

            ols_val = float(model.predict(X)[0])
            arima_val = self.arima_forecast_one(target)

            hybrid = 0.7*ols_val + 0.3*(arima_val if arima_val is not None else ols_val)

            out[target] = {
                "OLS": ols_val,
                "ARIMA": arima_val,
                "Hybrid": hybrid
            }

        return out

    # ----------------------------------------------------------
    # RUN PIPELINE
    # ----------------------------------------------------------
    def run(self):
        print("=== Training model ===")
        self.load_data()
        self.create_features()
        self.split_data()
        for t in ["maxtp","mintp"]:
            self.train_ols(t)
            self.train_arima(t)
            self.train_garch(t)
        print("=== Model Ready ===")

        # ==========================
        # Streamlit compatibility helpers
        # ==========================

    def run_pipeline(self):
        """Alias for Streamlit compatibility"""
        return self.run()

    def get_garch_volatility(self, target: str):
        model = self.garch_models.get(target)
        if model is None:
            return None
        try:
            return pd.Series(
                model.conditional_volatility,
                index=self.train.index
            )
        except Exception:
            return None

    def rolling_forecast_errors(self, target: str, window: int = 30):
        if target not in self.predictions:
            return None
        actual = self.predictions[target]["actual"]
        pred = self.predictions[target]["predicted"]
        err = actual - pred
        return np.sqrt((err ** 2).rolling(window=window, min_periods=1).mean())

    def get_metrics_df(self):
        rows = []
        for target in ["maxtp", "mintp"]:
            if target not in self.predictions:
                continue
            actual = self.predictions[target]["actual"]
            pred = self.predictions[target]["predicted"]
            rows.append({
                "target": target,
                "R2": r2_score(actual, pred),
                "MAE": mean_absolute_error(actual, pred),
                "RMSE": np.sqrt(mean_squared_error(actual, pred))
            })
        return pd.DataFrame(rows)

# %%
# ==============================================================================
# REPORTING FUNCTIONS (Independent - Can be called separately)
# ==============================================================================


def generate_data_summary(predictor: TemperaturePredictor):
    """Generate comprehensive data summary statistics"""
    print("\n" + "="*60)
    print("DATA SUMMARY STATISTICS")
    print("="*60)
    
    df = predictor.df
    
    # Basic statistics
    print("\nDataset Overview:")
    print(f"  Total Records: {len(df)}")
    print(f"  Date Range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    print(f"  Duration: {(df.index.max() - df.index.min()).days} days")
    
    # Summary statistics table
    print("\nDescriptive Statistics:")
    summary = df[['sun', 'soil', 'rain', 'gmin', 'wdsp', 'maxtp', 'mintp']].describe()
    print(summary.round(2))
    
    return summary



def plot_monthly_temperature_trends(predictor: TemperaturePredictor):
    """Plot monthly average temperatures"""
    df = predictor.df.copy()
    df['year'] = df.index.year
    df['month'] = df.index.month
    
    monthly_avg = df.groupby('month')[['maxtp', 'mintp']].mean()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = monthly_avg.index
    ax.plot(x, monthly_avg['maxtp'], marker='o', linewidth=2, markersize=8, label='Max Temperature', color='#e74c3c')
    ax.plot(x, monthly_avg['mintp'], marker='s', linewidth=2, markersize=8, label='Min Temperature', color='#3498db')
    
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
    ax.set_title('Monthly Average Temperature Trends', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('monthly_temperature_trends.png', dpi=300, bbox_inches='tight')
    #plt.show()()
    
    print("\n✔ Saved: monthly_temperature_trends.png")
    return monthly_avg


# NEW FUNCTION: Smoothed Temperature Trend
def plot_smoothed_temperature_trend(predictor: TemperaturePredictor):
    """Plot 12-month rolling average smoothed temperature trend"""
    # Get the original dataframe (before feature engineering removed rows)
    data = predictor.df.copy()
    
    # Monthly resample using index (since date is the index)
    data_month = data.resample("ME")[["maxtp", "mintp"]].mean()
    
    # Rolling smoothing
    data_month["maxtp_smooth"] = data_month["maxtp"].rolling(12).mean()
    data_month["mintp_smooth"] = data_month["mintp"].rolling(12).mean()
    
    # Plot
    plt.figure(figsize=(14, 5))
    plt.plot(data_month.index, data_month["maxtp_smooth"], 
             label="Max Temp (12-month Smooth)", linewidth=2.5, color='#e74c3c')
    plt.plot(data_month.index, data_month["mintp_smooth"], 
             label="Min Temp (12-month Smooth)", linewidth=2.5, color='#3498db')
    plt.title("Smoothed Temperature Trend (12-month Rolling Avg)", 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("Date", fontsize=12, fontweight='bold')
    plt.ylabel("Temperature (°C)", fontsize=12, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('smoothed_temperature_trend.png', dpi=300, bbox_inches='tight')
    #plt.show()()
    
    print("\n✔ Saved: smoothed_temperature_trend.png")
    return data_month


def plot_correlation_heatmap(predictor: TemperaturePredictor):
    """Plot correlation heatmap of features"""
    cols = ['sun', 'soil', 'rain', 'gmin', 'wdsp', 'maxtp', 'mintp']
    corr = predictor.df[cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Correlation Matrix of Weather Variables', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    #plt.show()()
    
    print("\n✔ Saved: correlation_heatmap.png")
    return corr


def display_ols_summary(predictor: TemperaturePredictor, target: str):
    """Display detailed OLS regression summary"""
    print("\n" + "="*60)
    print(f"OLS REGRESSION SUMMARY - {target.upper()}")
    print("="*60)
    
    model = predictor.ols_models[target]
    print(model.summary())
    
    # Extract key metrics
    print("\n" + "-"*60)
    print("KEY METRICS:")
    print("-"*60)
    print(f"  R-squared: {model.rsquared:.4f}")
    print(f"  Adjusted R-squared: {model.rsquared_adj:.4f}")
    print(f"  F-statistic: {model.fvalue:.2f}")
    print(f"  Prob (F-statistic): {model.f_pvalue:.2e}")
    print(f"  AIC: {model.aic:.2f}")
    print(f"  BIC: {model.bic:.2f}")
    
    return model.summary()


# NEW FUNCTION: Display ARIMA Summary
def display_arima_summary(predictor: TemperaturePredictor, target: str):
    """Display detailed ARIMA model summary"""
    print("\n" + "="*60)
    print(f"ARIMA MODEL SUMMARY - {target.upper()}")
    print("="*60)
    
    model = predictor.arima_models.get(target)
    
    if model is None:
        print(f"\n⚠ ARIMA model not available for {target.upper()}")
        return None
    
    print(model.summary())
    
    # Extract key metrics
    print("\n" + "-"*60)
    print("KEY METRICS:")
    print("-"*60)
    print(f"  AIC: {model.aic:.2f}")
    print(f"  BIC: {model.bic:.2f}")
    print(f"  HQIC: {model.hqic:.2f}")
    print(f"  Log Likelihood: {model.llf:.2f}")
    
    return model.summary()


# NEW FUNCTION: Display GARCH Summary
def display_garch_summary(predictor: TemperaturePredictor, target: str):
    """Display detailed GARCH model summary"""
    print("\n" + "="*60)
    print(f"GARCH MODEL SUMMARY - {target.upper()}")
    print("="*60)
    
    model = predictor.garch_models.get(target)
    
    if model is None:
        print(f"\n⚠ GARCH model not available for {target.upper()}")
        return None
    
    print(model.summary())
    
    # Extract key metrics
    print("\n" + "-"*60)
    print("KEY METRICS:")
    print("-"*60)
    print(f"  Log Likelihood: {model.loglikelihood:.2f}")
    print(f"  AIC: {model.aic:.2f}")
    print(f"  BIC: {model.bic:.2f}")
    
    return model.summary()


def plot_actual_vs_predicted(predictor: TemperaturePredictor, target: str):
    """Plot actual vs predicted values for test set"""
    actual = predictor.predictions[target]['actual']
    predicted = predictor.predictions[target]['predicted']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot
    ax1.scatter(actual, predicted, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Temperature (°C)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Temperature (°C)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Actual vs Predicted - {target.upper()}', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Time series comparison
    ax2.plot(actual.index, actual.values, label='Actual', linewidth=2, alpha=0.7)
    ax2.plot(predicted.index, predicted.values, label='Predicted', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Time Series - {target.upper()}', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'actual_vs_predicted_{target}.png', dpi=300, bbox_inches='tight')
    #plt.show()()
    
    print(f"\n✔ Saved: actual_vs_predicted_{target}.png")


def calculate_model_metrics(predictor: TemperaturePredictor):
    """Calculate and display comprehensive model performance metrics"""
    print("\n" + "="*60)
    print("MODEL PERFORMANCE METRICS")
    print("="*60)
    
    metrics_data = []
    
    for target in ['maxtp', 'mintp']:
        actual = predictor.predictions[target]['actual']
        predicted = predictor.predictions[target]['predicted']
        
        r2 = r2_score(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        metrics_data.append({
            'Target': target.upper(),
            'R² Score': f"{r2:.4f}",
            'MAE (°C)': f"{mae:.3f}",
            'RMSE (°C)': f"{rmse:.3f}",
            'MAPE (%)': f"{mape:.2f}"
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    print("\n", metrics_df.to_string(index=False))
    
    return metrics_df


def plot_residual_analysis(predictor: TemperaturePredictor, target: str):
    """Plot residual analysis for model diagnostics"""
    actual = predictor.predictions[target]['actual']
    predicted = predictor.predictions[target]['predicted']
    residuals = actual - predicted
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Residuals vs Predicted
    axes[0, 0].scatter(predicted, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Predicted Values (°C)', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Residuals (°C)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 1].set_xlabel('Residuals (°C)', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
    axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residuals over time
    axes[1, 1].plot(residuals.index, residuals.values, linewidth=1, alpha=0.7)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Date', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Residuals (°C)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Residuals Over Time', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    fig.suptitle(f'Residual Analysis - {target.upper()}', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f'residual_analysis_{target}.png', dpi=300, bbox_inches='tight')
    #plt.show()()
    
    print(f"\n✔ Saved: residual_analysis_{target}.png")


def plot_feature_importance(predictor: TemperaturePredictor, target: str):
    """Plot feature importance based on OLS coefficients"""
    model = predictor.ols_models[target]
    
    # Get coefficients (excluding constant)
    coef_df = pd.DataFrame({
        'Feature': model.params.index[1:],  # Skip constant
        'Coefficient': model.params.values[1:],
        'P-value': model.pvalues.values[1:]
    })
    
    # Sort by absolute coefficient value
    coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
    coef_df = coef_df.sort_values('Abs_Coefficient', ascending=True)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['green' if p < 0.05 else 'orange' for p in coef_df['P-value']]
    ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, edgecolor='black', linewidth=0.7)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Importance (OLS Coefficients) - {target.upper()}\nGreen: p<0.05, Orange: p≥0.05', 
                 fontsize=13, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f'feature_importance_{target}.png', dpi=300, bbox_inches='tight')
    #plt.show()()
    
    print(f"\n✔ Saved: feature_importance_{target}.png")
    
    return coef_df


# %%
def plot_garch_smoothed_volatility(predictor: TemperaturePredictor):
    """Plot 90-day smoothed GARCH conditional volatility for maxtp and mintp"""

    garch_max = predictor.garch_models.get("maxtp")
    garch_min = predictor.garch_models.get("mintp")

    if garch_max is None or garch_min is None:
        print("\n⚠ GARCH models not available. Train GARCH first.")
        return

    # Extract conditional volatility series (indexed by training dates)
    vol_max = pd.Series(garch_max.conditional_volatility, index=predictor.train.index)
    vol_min = pd.Series(garch_min.conditional_volatility, index=predictor.train.index)

    # 90-day rolling smoothing
    vol_max_smooth = vol_max.rolling(window=90).mean()
    vol_min_smooth = vol_min.rolling(window=90).mean()

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(vol_max_smooth, label="GARCH Volatility (MAX, 90-day smoothed)", color='blue')
    plt.plot(vol_min_smooth, label="GARCH Volatility (MIN, 90-day smoothed)", color='orange')
    
    plt.title("Smoothed Conditional Volatility from GARCH (90-day Rolling Mean)", fontsize=14, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Volatility", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("garch_smoothed_volatility.png", dpi=300)
    #plt.show()()

    print("\n✔ Saved: garch_smoothed_volatility.png")


# %%
def _fmt(v):
    return f"{v:.2f}°C" if v is not None else "N/A"



def get_float(prompt, min_val=None, max_val=None):
    """Safely get a float input, with optional min/max validation."""
    while True:
        try:
            value = float(input(prompt))

            if min_val is not None and value < min_val:
                print(f"Value must be ≥ {min_val}. Try again.")
                continue

            if max_val is not None and value > max_val:
                print(f"Value must be ≤ {max_val}. Try again.")
                continue

            return value
        
        except ValueError:
            print("Invalid input. Please enter a numeric value.")


def get_date(prompt):
    """Safely get a valid YYYY-MM-DD date."""
    while True:
        value = input(prompt).strip()
        try:
            dt = datetime.datetime.strptime(value, "%Y-%m-%d").date()
            return dt
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")






