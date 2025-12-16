# streamlit_app_final_sidebar.py
"""
Final dashboard with redesigned Overview (Hybrid style), Analytics and Forecast pages.
- Place Data.csv and optionally temp_predictor.py next to this file.
- Run: streamlit run streamlit_app_final_sidebar.py
"""

import os
import pickle
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import io
from PIL import Image

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Temperature Forecast Dashboard", layout="wide", initial_sidebar_state="expanded")

# ---------------------------
# CSS & Theme (Hybrid: subtle glow + clean)
# ---------------------------
CSS = """
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --accent: #4fb3ff;
  --accent-2: #7bd1ff;
  --glass: rgba(255,255,255,0.03);
  --card-border: rgba(255,255,255,0.05);
  --muted: #bcdcf8;
}
html, body, .stApp { font-family: 'Poppins', sans-serif; background: linear-gradient(180deg,#071225 0%, #03111a 100%); color: #eaf6ff; }
.header { display:flex; justify-content:space-between; align-items:center; gap:12px; }
.card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:14px; padding:14px; border:1px solid var(--card-border); box-shadow: 0 8px 30px rgba(0,0,0,0.5); }
.hero { display:flex; gap:18px; align-items:center; }
.hero-icon { width:110px; height:110px; border-radius:14px; display:flex; align-items:center; justify-content:center; font-size:44px; background: linear-gradient(135deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)); border:1px solid var(--card-border); }
.metric { color: var(--accent); font-weight:800; font-size:40px; }
.small { color: var(--muted); font-size:13px; }
.spark { height:72px; }
.hourly-carousel { display:flex; gap:10px; overflow-x:auto; padding:8px 2px; }
.hour-tile { min-width:96px; background: rgba(255,255,255,0.02); border-radius:10px; padding:8px; text-align:center; border:1px solid rgba(255,255,255,0.03); flex:0 0 auto; }
.day-row { display:flex; gap:10px; margin-top:10px; }
.day-card { flex:1; padding:12px; border-radius:10px; background: linear-gradient(180deg, rgba(255,255,255,0.015), rgba(255,255,255,0.01)); border:1px solid rgba(255,255,255,0.03); text-align:center; }
.stats-grid { display:grid; grid-template-columns: repeat(2, 1fr); gap:10px; margin-top:10px; }
.stButton>button, .stDownloadButton>button { background: var(--accent) !important; color:#042433 !important; border-radius:8px !important; padding:8px 12px !important; font-weight:700 !important; }
@media (max-width:900px) {
  .hero { flex-direction:column; align-items:flex-start; gap:8px; }
  .hour-tile { min-width:84px; }
  .stats-grid { grid-template-columns: 1fr; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------
# Load data
# ---------------------------
DATA_FILE = "Data.csv"
if not os.path.exists(DATA_FILE):
    st.error("Data.csv not found. Place Data.csv next to this script and reload.")
    st.stop()

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

try:
    df = load_data(DATA_FILE)
except Exception:
    st.error("Failed to load Data.csv.")
    st.exception(traceback.format_exc())
    st.stop()

# Validate required columns
if "maxtp" not in df.columns or "mintp" not in df.columns:
    st.error("Data.csv must contain 'maxtp' and 'mintp' columns.")
    st.stop()

# detect datetime column if present
def detect_datetime_col(data: pd.DataFrame) -> Optional[str]:
    candidates = ["date", "datetime", "time", "timestamp"]
    for c in data.columns:
        if c.lower() in candidates:
            return c
    for c in data.columns:
        try:
            pd.to_datetime(data[c].dropna().iloc[:10])
            return c
        except Exception:
            continue
    return None

dt_col = detect_datetime_col(df)
if dt_col:
    try:
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    except Exception:
        pass

# ---------------------------
# Optional: load TemperaturePredictor
# ---------------------------
HAS_TP = False
tp = None
try:
    from temp_predictor import TemperaturePredictor  # type: ignore
    HAS_TP = True
except Exception:
    HAS_TP = False

PICKLE = "models.pkl"
def load_cached_models(path=PICKLE):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None

if HAS_TP:
    cached = load_cached_models()
    if cached is not None:
        tp = cached

# ---------------------------
# Helpers
# ---------------------------
def weather_icon_for_row(row: pd.Series) -> str:
    try:
        rain = float(row.get("rain", 0) or 0)
        sun = float(row.get("sun", 0) or 0)
        wdsp = float(row.get("wdsp", 0) or 0)
        maxt = float(row.get("maxtp", 0) or 0)
    except Exception:
        rain = sun = wdsp = maxt = 0
    if rain > 5: return "⛈"
    if rain > 0.5: return "🌧"
    if maxt <= 0: return "❄️"
    if sun >= 6 and maxt >= 18: return "☀️"
    if sun >= 3: return "🌤"
    if wdsp > 12: return "🌫"
    return "🌥"

def safe_predict(tp_obj, params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return tp_obj.predict_from_dict(params)
    except Exception:
        return {}

def ensure_predictions_local(tp_obj, target: str) -> bool:
    if tp_obj is None:
        return False
    if getattr(tp_obj, "predictions", None) and target in tp_obj.predictions:
        return True
    try:
        if hasattr(tp_obj, "train_ols"):
            tp_obj.train_ols(target)
            return (target in getattr(tp_obj, "predictions", {}))
    except Exception:
        return False

def generate_24h(tp_obj, base_row: pd.Series):
    results = []
    last_maxt = base_row.get("maxtp", None)
    last_mint = base_row.get("mintp", None)
    base_dt = pd.to_datetime(base_row[dt_col]) if (dt_col in base_row and not pd.isna(base_row[dt_col])) else pd.Timestamp.now()
    for h in range(1,25):
        fdate = base_dt + pd.Timedelta(hours=h)
        params = {
            "sun": float(base_row.get("sun", 0) or 0),
            "soil": float(base_row.get("soil", 0) or 0),
            "rain": float(base_row.get("rain", 0) or 0),
            "gmin": float(base_row.get("gmin", 0) or 0),
            "wdsp": float(base_row.get("wdsp", 0) or 0),
            "maxtp_yesterday": last_maxt,
            "mintp_yesterday": last_mint,
            "forecast_date": pd.to_datetime(fdate)
        }
        if tp_obj is not None:
            out = safe_predict(tp_obj, params)
            ma = out.get("maxtp", {})
            mi = out.get("mintp", {})
            if isinstance(ma, dict) and "Hybrid" in ma:
                last_maxt = ma["Hybrid"]
            if isinstance(mi, dict) and "Hybrid" in mi:
                last_mint = mi["Hybrid"]
            results.append({"forecast_date": params["forecast_date"], "maxtp": ma, "mintp": mi})
        else:
            results.append({"forecast_date": params["forecast_date"], "maxtp": last_maxt, "mintp": last_mint})
    return results

def generate_7day(tp_obj, base_row: pd.Series):
    results = []
    last_maxt = base_row.get("maxtp", None)
    last_mint = base_row.get("mintp", None)
    base_dt = pd.to_datetime(base_row[dt_col]) if (dt_col in base_row and not pd.isna(base_row[dt_col])) else pd.Timestamp.now()
    for d in range(1,8):
        fdate = base_dt + pd.Timedelta(days=d)
        params = {
            "sun": float(base_row.get("sun", 0) or 0),
            "soil": float(base_row.get("soil", 0) or 0),
            "rain": float(base_row.get("rain", 0) or 0),
            "gmin": float(base_row.get("gmin", 0) or 0),
            "wdsp": float(base_row.get("wdsp", 0) or 0),
            "maxtp_yesterday": last_maxt,
            "mintp_yesterday": last_mint,
            "forecast_date": pd.to_datetime(fdate)
        }
        if tp_obj is not None:
            out = safe_predict(tp_obj, params)
            ma = out.get("maxtp", {})
            mi = out.get("mintp", {})
            if isinstance(ma, dict) and "Hybrid" in ma:
                last_maxt = ma["Hybrid"]
            if isinstance(mi, dict) and "Hybrid" in mi:
                last_mint = mi["Hybrid"]
            results.append({"forecast_date": params["forecast_date"], "maxtp": ma, "mintp": mi})
        else:
            results.append({"forecast_date": params["forecast_date"], "maxtp": last_maxt, "mintp": last_mint})
    return results

def safe_rolling_rmse_from_preds(actual: pd.Series, pred: pd.Series, window:int=30):
    try:
        comb = pd.concat([actual, pred], axis=1).dropna()
        comb.columns = ["actual","pred"]
        errs = comb["actual"] - comb["pred"]
        return np.sqrt((errs**2).rolling(window=window, min_periods=1).mean())
    except Exception:
        return None

# ---------------------------
# Sidebar navigation & controls
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Analytics", "Forecast & Downloads"])

st.sidebar.markdown("---")
st.sidebar.markdown("Model controls")
if HAS_TP:
    if st.sidebar.button("Train / Retrain models"):
        try:
            with st.spinner("Training models..."):
                tp = TemperaturePredictor(DATA_FILE)  # type: ignore
                tp.run_pipeline()
                try:
                    with open(PICKLE, "wb") as f:
                        pickle.dump(tp, f)
                except Exception:
                    pass
                st.sidebar.success("Training finished.")
        except Exception:
            st.sidebar.error("Training failed.")
            st.exception(traceback.format_exc())
    st.sidebar.checkbox("Use cached models if available", value=True)
else:
    st.sidebar.info("No temp_predictor.py detected — using fallbacks.")

st.sidebar.markdown("---")
st.sidebar.markdown("Theme")
theme = st.sidebar.selectbox("Theme", ["Dark", "Light"], index=0)
accent_choice = st.sidebar.selectbox("Accent", ["Blue","Purple","Orange","Green"], index=0)
accent_map = {"Blue":"#4fb3ff","Purple":"#b06cff","Orange":"#ff8a50","Green":"#2ecc71"}
accent_hex = accent_map.get(accent_choice, "#4fb3ff")
PLOTLY_TEMPLATE = "plotly_dark" if theme == "Dark" else "plotly_white"

# ---------------------------
# Page: Overview (enhanced)
# ---------------------------
if page == "Overview":
    st.header("Overview — Today & Forecast glance")
    #st.markdown("Beautiful summary, 14-day sparklines, hourly carousel and 7-day cards.")

    # two-column hero area: big hero + sparklines & stats
    col_hero, col_right = st.columns([1.4, 1])

    # get latest row
    try:
        latest_row = df.sort_values(by=dt_col).iloc[-1] if dt_col in df.columns else df.iloc[-1]
    except Exception:
        latest_row = df.iloc[-1]

    with col_hero:
        st.markdown('<div class="card hero">', unsafe_allow_html=True)
        # Hero: big icon + temps
        icon = weather_icon_for_row(latest_row)
        try:
            ma = float(latest_row.get("maxtp")) if not pd.isna(latest_row.get("maxtp")) else None
        except Exception:
            ma = None
        try:
            mi = float(latest_row.get("mintp")) if not pd.isna(latest_row.get("mintp")) else None
        except Exception:
            mi = None

        hero_html = f"""
        <div style="display:flex; gap:18px; align-items:center;">
          <div class="hero-icon" style="font-size:44px">{icon}</div>
          <div>
            <div style="font-size:14px; color:#bcdcf8">Today's Conditions</div>
            <div style="display:flex; gap:12px; align-items:end;">
              <div class="metric">{ma:.1f}°C</div>
              <div style="min-width:110px;">
                <div style="font-weight:700">{('Low: ' + (str(mi) + '°C')) if mi is not None else 'Low: n/a'}</div>
                <div class="small">Last update: {pd.to_datetime(latest_row[dt_col]).strftime('%Y-%m-%d %H:%M') if dt_col in latest_row and not pd.isna(latest_row[dt_col]) else 'n/a'}</div>
              </div>
            </div>
          </div>
        </div>
        """
        st.markdown(hero_html, unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # Hourly carousel (24 -> show horizontal scroll)
        st.markdown("<div style='font-weight:700; margin-bottom:6px'>24-hour forecast (carousel)</div>", unsafe_allow_html=True)
        hourly_html = '<div class="hourly-carousel">'
        # generate using tp if available else fallback to last 24 values
        hourly = []
        if tp is not None:
            try:
                hourly = generate_24h(tp, latest_row)
            except Exception:
                hourly = []
        if hourly:
            for r in hourly:
                fdt = pd.to_datetime(r["forecast_date"])
                label = fdt.strftime("%a %I %p")
                ma_val = None
                if isinstance(r["maxtp"], dict) and "Hybrid" in r["maxtp"]:
                    ma_val = r["maxtp"]["Hybrid"]
                elif isinstance(r["maxtp"], (int, float)):
                    ma_val = r["maxtp"]
                icon2 = "🌤"
                try:
                    icon2 = "☀️" if ma_val is not None and float(ma_val) >= 18 else icon2
                except Exception:
                    pass
                hourly_html += f'<div class="hour-tile"><div style="font-weight:700">{label}</div><div style="font-size:18px; margin-top:6px">{icon2}</div><div style="margin-top:6px">{("" if ma_val is None else f"{ma_val:.1f}°")}</div></div>'
        else:
            # fallback to last 24 observations (or fewer)
            sample = df.sort_values(by=dt_col).tail(24) if dt_col in df.columns else df.tail(12)
            for _, r in sample.iterrows():
                label = ""
                if dt_col in r and not pd.isna(r.get(dt_col, None)):
                    try:
                        label = pd.to_datetime(r[dt_col]).strftime("%a %I %p")
                    except Exception:
                        label = str(r[dt_col])
                v = r.get("maxtp", "")
                icon2 = weather_icon_for_row(r)
                hourly_html += f'<div class="hour-tile"><div style="font-weight:700">{label}</div><div style="font-size:18px; margin-top:6px">{icon2}</div><div style="margin-top:6px">{v if pd.notna(v) else ""}</div></div>'
        hourly_html += "</div>"
        st.markdown(hourly_html, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<div style='display:flex; justify-content:space-between; align-items:center;'><div style='font-weight:700'>Trends & stats</div><div class='small'>Last 14 days</div></div>", unsafe_allow_html=True)

        # small sparklines for maxtp and mintp
        def sparkline(series, title):
            fig = px.line(series, height=90, template=PLOTLY_TEMPLATE)
            fig.update_traces(line=dict(color=accent_hex), showlegend=False)
            fig.update_layout(margin=dict(l=0,r=0,t=10,b=10), xaxis=dict(visible=False), yaxis=dict(visible=False))
            st.markdown(f"<div style='font-weight:700'>{title}</div>", unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)

        last_n = 14
        try:
            if dt_col in df.columns:
                smax = df.sort_values(by=dt_col).set_index(dt_col)["maxtp"].dropna().astype(float).tail(last_n)
                smin = df.sort_values(by=dt_col).set_index(dt_col)["mintp"].dropna().astype(float).tail(last_n)
            else:
                smax = df["maxtp"].dropna().astype(float).tail(last_n)
                smin = df["mintp"].dropna().astype(float).tail(last_n)
        except Exception:
            smax = df["maxtp"].dropna().astype(float).tail(last_n) if "maxtp" in df.columns else pd.Series([])
            smin = df["mintp"].dropna().astype(float).tail(last_n) if "mintp" in df.columns else pd.Series([])

        sparkline(smax, "Max temp (last 14 days)")
        sparkline(smin, "Min temp (last 14 days)")

        # Quick stats grid
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='stats-grid'>", unsafe_allow_html=True)
        # dataset size
        try:
            rows = df.shape[0]
            missing = df.isnull().sum().sum()
            temp_range = df["maxtp"].dropna().astype(float).max() - df["maxtp"].dropna().astype(float).min()
        except Exception:
            rows = df.shape[0]
            missing = df.isnull().sum().sum()
            temp_range = None

        # --- fix: compute display string before formatting ---
        temp_range_str = f"{temp_range:.1f}" if (temp_range is not None and not pd.isna(temp_range)) else "n/a"
        last_date_str = pd.to_datetime(df[dt_col]).max().strftime('%Y-%m-%d') if dt_col in df.columns else "n/a"

        st.markdown(f"<div style='padding:10px; border-radius:8px; background:rgba(255,255,255,0.01)'><div style='font-weight:700'>{rows}</div><div class='small'>Rows</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='padding:10px; border-radius:8px; background:rgba(255,255,255,0.01)'><div style='font-weight:700'>{missing}</div><div class='small'>Missing values</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='padding:10px; border-radius:8px; background:rgba(255,255,255,0.01)'><div style='font-weight:700'>{temp_range_str}</div><div class='small'>Max range (maxtp)</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='padding:10px; border-radius:8px; background:rgba(255,255,255,0.01)'><div style='font-weight:700'>{last_date_str}</div><div class='small'>Last date</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 7-day modern cards (full width)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700'>7-day forecast</div>", unsafe_allow_html=True)
    # generate using tp if possible else show last 7 days averages
    seven_html = '<div class="day-row">'
    try:
        if tp is not None:
            seven = generate_7day(tp, latest_row)
            for d in seven:
                fdt = pd.to_datetime(d["forecast_date"]).date()
                ma_val = None
                if isinstance(d["maxtp"], dict) and "Hybrid" in d["maxtp"]:
                    ma_val = d["maxtp"]["Hybrid"]
                elif isinstance(d["maxtp"], (int,float)):
                    ma_val = d["maxtp"]
                icon = "🌤"
                if ma_val is not None and ma_val >= 18:
                    icon = "☀️"
                seven_html += f'<div class="day-card"><div style="font-weight:700">{fdt.strftime("%a %d")}</div><div style="font-size:20px; margin-top:6px">{icon}</div><div style="margin-top:8px; font-weight:700">{("" if ma_val is None else f"{ma_val:.1f}°")}</div></div>'
        else:
            if dt_col in df.columns:
                tmp = df.sort_values(by=dt_col).copy()
                tmp["date_only"] = pd.to_datetime(tmp[dt_col]).dt.date
                last_dates = sorted(tmp["date_only"].unique())[-7:]
                for d in last_dates:
                    sub = tmp[tmp["date_only"] == d]
                    avg = sub["maxtp"].dropna().astype(float).mean() if "maxtp" in sub.columns else None
                    avg_s = f"{avg:.1f}°" if avg is not None and not pd.isna(avg) else "n/a"
                    seven_html += f'<div class="day-card"><div style="font-weight:700">{d.strftime("%a %d")}</div><div style="font-size:20px; margin-top:6px">🌤</div><div style="margin-top:8px; font-weight:700">{avg_s}</div></div>'
            else:
                sample = df.tail(7)
                for _, r in sample.iterrows():
                    v = r.get("maxtp","n/a")
                    seven_html += f'<div class="day-card"><div style="font-weight:700">Day</div><div style="font-size:20px; margin-top:6px">🌤</div><div style="margin-top:8px; font-weight:700">{v}</div></div>'
    except Exception:
        seven_html += '<div style="color:#f99">7-day generation failed</div>'
    seven_html += "</div>"
    st.markdown(seven_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Page: Analytics (full-width charts + model summaries)
# ---------------------------
elif page == "Analytics":
    st.header("Analytics — Data, Diagnostics & Models")
    #st.markdown("Full-width interactive charts and organized model summaries.")

    tab_data, tab_diag, tab_models = st.tabs(["Data Preview", "Diagnostics", "Model Summaries"])

    with tab_data:
        st.subheader("Data preview")
        st.dataframe(df.head(200))
        info = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isnull().sum()})
        st.table(info)

    with tab_diag:
        st.subheader("Diagnostics (full-width)")
        target = st.selectbox("Target", ["maxtp","mintp"], index=0)
        # time series
        if st.button("Show time series"):
            try:
                if dt_col in df.columns:
                    s = df.sort_values(by=dt_col).set_index(dt_col)[target].dropna().astype(float)
                    fig = px.line(s, title=f"{target} time series", template=PLOTLY_TEMPLATE)
                    fig.update_traces(line=dict(color=accent_hex))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Datetime column required.")
            except Exception:
                st.error("Time series failed.")
                st.exception(traceback.format_exc())

        # actual vs predicted
        if st.button("Actual vs Predicted"):
            try:
                ok = ensure_predictions_local(tp, target) if tp else False
                if ok:
                    act = tp.predictions[target]["actual"]
                    pr = tp.predictions[target]["predicted"]
                    dfp = pd.concat([act, pr], axis=1).dropna()
                    dfp.columns = ["actual","pred"]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=dfp.index, y=dfp["actual"], name="Actual", mode="lines"))
                    fig.add_trace(go.Scatter(x=dfp.index, y=dfp["pred"], name="Predicted", mode="lines"))
                    fig.update_layout(title=f"Actual vs Predicted ({target})", template=PLOTLY_TEMPLATE)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    if dt_col in df.columns:
                        s = df.sort_values(by=dt_col).set_index(dt_col)[target].dropna().astype(float)
                        p = s.shift(1).dropna()
                        s2 = s.loc[p.index]
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=s2.index, y=s2.values, name="Actual", mode="lines"))
                        fig.add_trace(go.Scatter(x=p.index, y=p.values, name="Prev-day", mode="lines"))
                        fig.update_layout(title=f"Actual vs Prev-day ({target})", template=PLOTLY_TEMPLATE)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No predictions and no datetime column for fallback.")
            except Exception:
                st.error("Actual vs Predicted failed.")
                st.exception(traceback.format_exc())

        # residuals histogram
        if st.button("Residuals histogram"):
            try:
                ok = ensure_predictions_local(tp, target) if tp else False
                if ok:
                    act = tp.predictions[target]["actual"]; pr = tp.predictions[target]["predicted"]
                    resid = (act - pr).dropna()
                else:
                    if dt_col in df.columns:
                        s = df.sort_values(by=dt_col).set_index(dt_col)[target].dropna().astype(float)
                        p = s.shift(1).dropna()
                        resid = (s.loc[p.index] - p).dropna()
                    else:
                        resid = None
                if resid is None or resid.empty:
                    st.warning("No residuals to plot.")
                else:
                    fig = px.histogram(resid, nbins=40, title="Residuals", template=PLOTLY_TEMPLATE)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.error("Residuals failed.")
                st.exception(traceback.format_exc())

        # ACF/PACF
        if st.button("ACF & PACF"):
            try:
                ok = ensure_predictions_local(tp, target) if tp else False
                if ok:
                    act = tp.predictions[target]["actual"]; pr = tp.predictions[target]["predicted"]
                    resid = (act - pr).dropna()
                else:
                    if dt_col in df.columns:
                        s = df.sort_values(by=dt_col).set_index(dt_col)[target].dropna().astype(float)
                        p = s.shift(1).dropna()
                        resid = (s.loc[p.index] - p).dropna()
                    else:
                        resid = None
                if resid is None or resid.empty:
                    st.warning("No residuals for ACF/PACF.")
                else:
                    fig1 = plt.figure(figsize=(12,3))
                    plot_acf(resid, lags=40, ax=fig1.add_subplot(111))
                    buf = io.BytesIO(); fig1.tight_layout(); fig1.savefig(buf, format="png"); buf.seek(0)
                    st.image(buf, caption="ACF", use_column_width=True); plt.close(fig1)
                    fig2 = plt.figure(figsize=(12,3))
                    plot_pacf(resid, lags=40, ax=fig2.add_subplot(111), method="ywm")
                    buf2 = io.BytesIO(); fig2.tight_layout(); fig2.savefig(buf2, format="png"); buf2.seek(0)
                    st.image(buf2, caption="PACF", use_column_width=True); plt.close(fig2)
            except Exception:
                st.error("ACF/PACF failed.")
                st.exception(traceback.format_exc())

        # rolling RMSE
        if st.button("30-day rolling RMSE"):
            try:
                used = False
                if tp is not None and hasattr(tp, "rolling_forecast_errors"):
                    try:
                        roll = tp.rolling_forecast_errors(target, window=30)
                        if roll is not None:
                            fig = px.line(x=roll.index, y=roll.values, title="30-day rolling RMSE", template=PLOTLY_TEMPLATE)
                            st.plotly_chart(fig, use_container_width=True)
                            used = True
                    except Exception:
                        pass
                if not used:
                    if tp is not None and getattr(tp, "predictions", None) and target in tp.predictions:
                        act = tp.predictions[target]["actual"]; pr = tp.predictions[target]["predicted"]
                        roll = safe_rolling_rmse_from_preds(act, pr, window=30)
                        if roll is not None:
                            fig = px.line(x=roll.index, y=roll.values, title="30-day rolling RMSE", template=PLOTLY_TEMPLATE)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        if dt_col in df.columns:
                            s = df.sort_values(by=dt_col).set_index(dt_col)[target].dropna().astype(float)
                            if len(s) < 2:
                                st.warning("Not enough data for naive RMSE fallback.")
                            else:
                                p = s.shift(1)
                                comb = pd.concat([s,p], axis=1).dropna(); comb.columns=["actual","pred"]
                                errs = comb["actual"] - comb["pred"]
                                roll = np.sqrt((errs**2).rolling(window=30, min_periods=1).mean())
                                fig = px.line(x=roll.index, y=roll.values, title="30-day rolling RMSE (naive)", template=PLOTLY_TEMPLATE)
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No data for RMSE fallback.")
            except Exception:
                st.error("Rolling RMSE failed.")
                st.exception(traceback.format_exc())

        # volatility
        if st.button("GARCH volatility"):
            try:
                shown = False
                if tp is not None and hasattr(tp, "get_garch_volatility"):
                    try:
                        vol = tp.get_garch_volatility(target)
                        if vol is not None:
                            fig = px.line(x=vol.index, y=vol.values, title="GARCH conditional volatility", template=PLOTLY_TEMPLATE)
                            st.plotly_chart(fig, use_container_width=True)
                            shown = True
                    except Exception:
                        pass
                if not shown:
                    if tp is not None and getattr(tp, "predictions", None) and target in tp.predictions:
                        act = tp.predictions[target]["actual"]; pr = tp.predictions[target]["predicted"]
                        resid = (act - pr).dropna()
                        vol_proxy = resid.rolling(window=30, min_periods=5).std()
                        fig = px.line(x=vol_proxy.index, y=vol_proxy.values, title="Fallback volatility (30-day rolling std)", template=PLOTLY_TEMPLATE)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        if dt_col in df.columns:
                            s = df.sort_values(by=dt_col).set_index(dt_col)[target].dropna().astype(float)
                            vol_proxy = s.rolling(window=30, min_periods=5).std()
                            fig = px.line(x=vol_proxy.index, y=vol_proxy.values, title="Fallback volatility (30-day rolling std)", template=PLOTLY_TEMPLATE)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No data to compute volatility fallback.")
            except Exception:
                st.error("Volatility failed.")
                st.exception(traceback.format_exc())

    with tab_models:
        st.subheader("Model Summaries (tabbed)")
        ols_tab, arima_tab, garch_tab = st.tabs(["OLS", "ARIMA", "GARCH"])

        with ols_tab:
            st.markdown("#### OLS summaries")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Show maxtp OLS summary"):
                    try:
                        if tp is not None and hasattr(tp, "ols_models") and "maxtp" in tp.ols_models:
                            st.text(tp.ols_models["maxtp"].summary().as_text())
                        else:
                            st.warning("maxtp OLS not available.")
                    except Exception:
                        st.error("Failed to show maxtp OLS.")
                        st.exception(traceback.format_exc())
            with c2:
                if st.button("Show mintp OLS summary"):
                    try:
                        if tp is not None and hasattr(tp, "ols_models") and "mintp" in tp.ols_models:
                            st.text(tp.ols_models["mintp"].summary().as_text())
                        else:
                            st.warning("mintp OLS not available.")
                    except Exception:
                        st.error("Failed to show mintp OLS.")
                        st.exception(traceback.format_exc())

        with arima_tab:
            st.markdown("#### ARIMA summaries")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Show maxtp ARIMA summary"):
                    try:
                        if tp is not None and hasattr(tp, "arima_models") and "maxtp" in tp.arima_models:
                            st.text(tp.arima_models["maxtp"].summary().as_text())
                        else:
                            st.warning("maxtp ARIMA not available.")
                    except Exception:
                        st.error("Failed to show maxtp ARIMA.")
                        st.exception(traceback.format_exc())
            with c2:
                if st.button("Show mintp ARIMA summary"):
                    try:
                        if tp is not None and hasattr(tp, "arima_models") and "mintp" in tp.arima_models:
                            st.text(tp.arima_models["mintp"].summary().as_text())
                        else:
                            st.warning("mintp ARIMA not available.")
                    except Exception:
                        st.error("Failed to show mintp ARIMA.")
                        st.exception(traceback.format_exc())

        with garch_tab:
            st.markdown("#### GARCH summaries")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Show maxtp GARCH summary"):
                    try:
                        if tp is not None and hasattr(tp, "garch_models") and "maxtp" in tp.garch_models:
                            gm = tp.garch_models["maxtp"]
                            try:
                                st.text(gm.summary().as_text())
                            except Exception:
                                st.text(str(gm))
                        else:
                            st.warning("maxtp GARCH not available.")
                    except Exception:
                        st.error("Failed to show maxtp GARCH.")
                        st.exception(traceback.format_exc())
            with c2:
                if st.button("Show mintp GARCH summary"):
                    try:
                        if tp is not None and hasattr(tp, "garch_models") and "mintp" in tp.garch_models:
                            gm = tp.garch_models["mintp"]
                            try:
                                st.text(gm.summary().as_text())
                            except Exception:
                                st.text(str(gm))
                        else:
                            st.warning("mintp GARCH not available.")
                    except Exception:
                        st.error("Failed to show mintp GARCH.")
                        st.exception(traceback.format_exc())

# ---------------------------
# Page: Forecast & Downloads
# ---------------------------
elif page == "Forecast & Downloads":
    st.header("Forecast & Downloads")
    st.markdown("Single-date forecast, metrics and model artifacts.")

    with st.form("forecast_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            sun = st.number_input("sun (hours)", value=float(df["sun"].dropna().iloc[-1]) if "sun" in df.columns and not df["sun"].dropna().empty else 0.0)
            soil = st.number_input("soil", value=float(df["soil"].dropna().iloc[-1]) if "soil" in df.columns and not df["soil"].dropna().empty else 0.0)
        with c2:
            rain = st.number_input("rain (mm)", value=float(df["rain"].dropna().iloc[-1]) if "rain" in df.columns and not df["rain"].dropna().empty else 0.0)
            gmin = st.number_input("gmin", value=float(df["gmin"].dropna().iloc[-1]) if "gmin" in df.columns and not df["gmin"].dropna().empty else 0.0)
        with c3:
            wdsp = st.number_input("wdsp", value=float(df["wdsp"].dropna().iloc[-1]) if "wdsp" in df.columns and not df["wdsp"].dropna().empty else 0.0)
            maxtp_y = st.number_input("maxtp yesterday", value=float(df["maxtp"].dropna().iloc[-1]) if not df["maxtp"].dropna().empty else 0.0)
        forecast_date = st.date_input("Forecast date", value=datetime.now().date() + timedelta(days=1))
        submitted = st.form_submit_button("Get forecast")
        if submitted:
            params = {
                "sun": sun, "soil": soil, "rain": rain,
                "gmin": gmin, "wdsp": wdsp,
                "maxtp_yesterday": maxtp_y,
                "mintp_yesterday": float(df["mintp"].dropna().iloc[-1]) if not df["mintp"].dropna().empty else None,
                "forecast_date": pd.to_datetime(forecast_date)
            }
            if tp is not None:
                try:
                    out = tp.predict_from_dict(params)
                    st.markdown("**Model outputs**")
                    st.json(out)
                    try:
                        st.metric("Predicted MAX (Hybrid)", f"{out['maxtp']['Hybrid']:.2f} °C")
                        st.metric("Predicted MIN (Hybrid)", f"{out['mintp']['Hybrid']:.2f} °C")
                    except Exception:
                        pass
                except Exception:
                    st.error("Prediction failed.")
                    st.exception(traceback.format_exc())
            else:
                st.warning("Model not available. Add temp_predictor.py and retrain to enable predictions.")

    st.markdown("---")
    st.subheader("Metrics & artifacts")
    if tp is not None and hasattr(tp, "get_metrics_df"):
        try:
            metrics = tp.get_metrics_df()
            st.dataframe(metrics)
        except Exception:
            st.warning("get_metrics_df failed.")
            st.exception(traceback.format_exc())
    else:
        st.info("Metrics not available (tp missing).")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Download models.pkl"):
            try:
                if tp is not None:
                    with open(PICKLE, "wb") as f:
                        pickle.dump(tp, f)
                    with open(PICKLE, "rb") as f:
                        st.download_button("Download models.pkl", data=f, file_name="models.pkl")
                else:
                    st.warning("No trained model to download.")
            except Exception:
                st.error("Failed saving models.pkl")
                st.exception(traceback.format_exc())
    with c2:
        if st.button("Download predictions CSV"):
            try:
                rows = []
                if tp is not None and getattr(tp, "predictions", None):
                    for t in ["maxtp","mintp"]:
                        if t in tp.predictions:
                            a = tp.predictions[t]["actual"]; p = tp.predictions[t]["predicted"]
                            dfp = pd.DataFrame({"date": a.index, f"{t}_actual": a.values, f"{t}_predicted": p.values})
                            rows.append(dfp)
                if rows:
                    out = pd.concat(rows, axis=1)
                    csv_bytes = out.to_csv(index=False).encode("utf-8")
                    st.download_button("Download predictions.csv", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
                else:
                    st.warning("No predictions available.")
            except Exception:
                st.error("Failed creating CSV.")
                st.exception(traceback.format_exc())

    st.markdown("---")
    st.subheader("Auto 24-hour & 7-day forecast (latest row)")
    try:
        latest_row = df.sort_values(by=dt_col).iloc[-1] if dt_col in df.columns else df.iloc[-1]
    except Exception:
        latest_row = df.iloc[-1]

    if st.button("Generate 24-hour forecast"):
        try:
            res24 = generate_24h(tp, latest_row)
            rows = []
            for r in res24:
                dtv = pd.to_datetime(r["forecast_date"])
                ma = r["maxtp"]
                ma_val = None
                if isinstance(ma, dict) and "Hybrid" in ma:
                    ma_val = ma["Hybrid"]
                elif isinstance(ma, (int,float)):
                    ma_val = ma
                rows.append({"datetime": dtv, "maxtp": ma_val})
            st.dataframe(pd.DataFrame(rows))
        except Exception:
            st.error("24-hour forecast failed.")
            st.exception(traceback.format_exc())

    if st.button("Generate 7-day forecast"):
        try:
            res7 = generate_7day(tp, latest_row)
            rows = []
            for r in res7:
                dtv = pd.to_datetime(r["forecast_date"]).date()
                ma = r["maxtp"]
                ma_val = None
                if isinstance(ma, dict) and "Hybrid" in ma:
                    ma_val = ma["Hybrid"]
                elif isinstance(ma, (int,float)):
                    ma_val = ma
                rows.append({"date": dtv, "maxtp": ma_val})
            st.dataframe(pd.DataFrame(rows))
        except Exception:
            st.error("7-day forecast failed.")
            st.exception(traceback.format_exc())

# ---------------------------
# Footer: data sample
# ---------------------------
st.markdown("---")
st.markdown("<div style='color:#cfe9ff; font-weight:700'>Data sample</div>", unsafe_allow_html=True)
st.dataframe(df.head(50))

