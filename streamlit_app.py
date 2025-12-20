import os
import pickle
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import io
from pathlib import Path
import base64
import textwrap

# Page config
st.set_page_config(
    page_title="Temperature Forecast Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌡️"
)

# Enhanced CSS with professional styling
CSS = textwrap.dedent("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">
    <style>
    :root {
      --primary: #2E86DE;
      --secondary: #5F27CD;
      --success: #00D2D3;
      --warning: #FFA801;
      --danger: #EE5A6F;
      --text-light: #E8F4F8;
      --card-bg: rgba(255,255,255,0.03);
      --border: rgba(255,255,255,0.08);
    }
    html, body, .stApp {
      font-family: 'Inter', sans-serif;
      color: var(--text-light);
    }
    .main-header {
      background: linear-gradient(135deg, var(--primary), var(--secondary));
      padding: 2rem;
      border-radius: 16px;
      margin-bottom: 2rem;
      box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    .main-header h1 {
      margin: 0;
      font-size: 2.5rem;
      font-weight: 800;
      color: white;
    }
    .main-header p {
      margin: 0.5rem 0 0;
      font-size: 1.1rem;
      opacity: 0.9;
      color: white;
    }
    .metric-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.5rem;
      backdrop-filter: blur(10px);
      box-shadow: 0 8px 32px rgba(0,0,0,0.2);
      transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
      transform: translateY(-4px);
      box-shadow: 0 12px 48px rgba(0,0,0,0.3);
    }
    .metric-value {
      font-size: 2.5rem;
      font-weight: 800;
      background: linear-gradient(135deg, var(--primary), var(--success));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin: 0.5rem 0;
    }
    .metric-label {
      font-size: 0.9rem;
      color: var(--text-light);
      opacity: 0.7;
      text-transform: uppercase;
      letter-spacing: 1px;
      font-weight: 600;
    }
    .metric-delta {
      font-size: 0.85rem;
      margin-top: 0.5rem;
      padding: 0.25rem 0.5rem;
      border-radius: 6px;
      display: inline-block;
    }
    .delta-positive {
      background: rgba(0,210,211,0.15);
      color: var(--success);
    }
    .delta-negative {
      background: rgba(238,90,111,0.15);
      color: var(--danger);
    }
    .info-card {
      background: linear-gradient(135deg, rgba(46,134,222,0.1), rgba(95,39,205,0.1));
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.5rem;
      margin: 1rem 0;
    }
    .info-card h3 {
      margin: 0 0 1rem 0;
      font-size: 1.3rem;
      font-weight: 700;
    }
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
      margin: 1.5rem 0;
    }
    .stat-item {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1.25rem;
      text-align: center;
    }
    .stat-value {
      font-size: 2rem;
      font-weight: 700;
      color: var(--primary);
    }
    .stat-label {
      font-size: 0.85rem;
      opacity: 0.7;
      margin-top: 0.5rem;
    }
    .section-header {
      font-size: 1.3rem;
      font-weight: 700;
      margin: 2rem 0 1rem 0;
      padding-bottom: 0.5rem;
      border-bottom: 2px solid var(--primary);
    }
    .model-summary {
      background: linear-gradient(135deg, rgba(46,134,222,0.05), rgba(95,39,205,0.05));
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.5rem;
      margin: 1rem 0;
    }
    .model-summary h4 {
      margin: 0 0 1rem 0;
      color: var(--primary);
      font-weight: 700;
    }
    .stButton>button {
      background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
      color: white !important;
      border: none !important;
      border-radius: 8px !important;
      padding: 0.75rem 2rem !important;
      font-weight: 600 !important;
      transition: transform 0.2s !important;
    }
    .stButton>button:hover {
      transform: scale(1.05) !important;
      box-shadow: 0 8px 24px rgba(46,134,222,0.4) !important;
    }
    .stTabs [data-baseweb="tab-list"] {
      gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
      background: var(--card-bg);
      border-radius: 8px 8px 0 0;
      padding: 0.75rem 1.5rem;
      font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
      background: linear-gradient(135deg, var(--primary), var(--secondary));
      color: white;
    }
    .forecast-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1rem;
      text-align: center;
      min-width: 120px;
    }
    .forecast-icon {
      font-size: 2.5rem;
      margin: 0.5rem 0;
    }
    .forecast-temp {
      font-size: 1.8rem;
      font-weight: 700;
      color: var(--primary);
    }
    .forecast-date {
      font-size: 0.9rem;
      opacity: 0.7;
      font-weight: 600;
    }
    section[data-testid="stSidebar"] {
      background: rgba(0,0,0,0.6);
      backdrop-filter: blur(20px);
    }
    section[data-testid="stSidebar"] .stRadio > label {
      font-weight: 600;
      font-size: 1rem;
    }
    .dataframe {
      border-radius: 8px;
      overflow: hidden;
    }
    @media (max-width: 768px) {
      .main-header h1 { font-size: 1.8rem; }
      .stats-grid { grid-template-columns: 1fr; }
    }
    </style>
""")
st.markdown(CSS, unsafe_allow_html=True)

def set_background(image_file):
    """Set background image if available"""
    if not os.path.exists(image_file):
        return
    try:
        img_bytes = Path(image_file).read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        bg_css = textwrap.dedent(f"""
            <style>
            [data-testid="stAppViewContainer"] {{
                background-image: url("data:image/jpg;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            .block-container {{
                background-color: rgba(0, 0, 0, 0.55);
                backdrop-filter: blur(2px);
                border-radius: 18px;
            }}
            </style>
            """)
        st.markdown(bg_css, unsafe_allow_html=True)
    except Exception:
        pass

# Try to set background
set_background("background.jpg")

# Load data
DATA_FILE = "Data.csv"
if not os.path.exists(DATA_FILE):
    st.error("⚠️ Data.csv not found. Please place Data.csv in the same directory.")
    st.stop()

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

try:
    df = load_data(DATA_FILE)
except Exception as e:
    st.error("Failed to load Data.csv.")
    st.exception(e)
    st.stop()

# Validate required columns
required_cols = ["maxtp", "mintp"]
if not all(col in df.columns for col in required_cols):
    st.error("Data.csv must contain 'maxtp' and 'mintp' columns.")
    st.stop()

# Detect datetime column
def detect_datetime_col(data: pd.DataFrame) -> Optional[str]:
    candidates = ["date", "datetime", "time", "timestamp"]
    for c in data.columns:
        if c.lower() in candidates:
            return c
    return None

dt_col = detect_datetime_col(df)
if dt_col:
    try:
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
        df = df.sort_values(by=dt_col).reset_index(drop=True)
    except Exception:
        pass
else:
    st.warning("No datetime column detected. Some features may be limited.")

# Load temperature predictor if available
HAS_TP = False
tp = None
try:
    from temp_predictor import TemperaturePredictor
    HAS_TP = True
    
    PICKLE = "models.pkl"
    if os.path.exists(PICKLE):
        try:
            with open(PICKLE, "rb") as f:
                tp = pickle.load(f)
        except Exception:
            pass
except Exception:
    pass

# Helper functions
def weather_icon_for_temp(temp, rain=0):
    """Get weather icon based on temperature and rain"""
    if rain > 5: return "⛈️"
    if rain > 0.5: return "🌧️"
    if temp <= 0: return "❄️"
    if temp >= 25: return "☀️"
    if temp >= 18: return "🌤️"
    return "🌥️"

def calculate_statistics(series):
    """Calculate comprehensive statistics"""
    return {
        "mean": series.mean(),
        "std": series.std(),
        "min": series.min(),
        "max": series.max(),
        "median": series.median(),
        "q25": series.quantile(0.25),
        "q75": series.quantile(0.75)
    }

# Sidebar
st.sidebar.markdown("### 🌡️ Temperature Forecast")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Executive Summary", "📈 Data Analysis", "🔬 Model Diagnostics", "🔮 Forecasting", "📥 Export & Reports"],
    label_visibility="collapsed"
)
st.sidebar.markdown("---")

# Model controls
if HAS_TP:
    st.sidebar.markdown("### 🤖 Model Controls")
    if st.sidebar.button("🔄 Train/Retrain Models"):
        with st.spinner("Training models... This may take a minute."):
            try:
                tp = TemperaturePredictor(DATA_FILE)
                tp.run()
                with open("models.pkl", "wb") as f:
                    pickle.dump(tp, f)
                st.sidebar.success("✅ Models trained successfully!")
                st.rerun()
            except Exception as e:
                st.sidebar.error("❌ Training failed")
                st.sidebar.exception(e)
else:
    st.sidebar.info("💡 Place temp_predictor.py in the directory to enable model training")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Settings")
theme = st.sidebar.selectbox("Chart Theme", ["plotly_dark", "plotly_white", "seaborn"], index=0)

# Main content
if page == "📊 Executive Summary":
    # Header
    header_html = textwrap.dedent("""
        <div class="main-header">
            <h1>🌡️ Temperature Forecast Dashboard</h1>
            <p>Advanced Time Series Analysis & Prediction System</p>
        </div>
        """)
    st.markdown(header_html, unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        latest_max = df["maxtp"].dropna().iloc[-1] if not df["maxtp"].dropna().empty else 0
        prev_max = df["maxtp"].dropna().iloc[-2] if len(df["maxtp"].dropna()) > 1 else latest_max
        delta_max = latest_max - prev_max
        
        metric_html = textwrap.dedent(f"""
            <div class="metric-card">
                <div class="metric-label">Current Max Temp</div>
                <div class="metric-value">{latest_max:.1f}°C</div>
                <div class="metric-delta {'delta-positive' if delta_max >= 0 else 'delta-negative'}">
                    {'↑' if delta_max >= 0 else '↓'} {abs(delta_max):.1f}°C
                </div>
            </div>
            """)
        st.markdown(metric_html, unsafe_allow_html=True)
    
    with col2:
        latest_min = df["mintp"].dropna().iloc[-1] if not df["mintp"].dropna().empty else 0
        prev_min = df["mintp"].dropna().iloc[-2] if len(df["mintp"].dropna()) > 1 else latest_min
        delta_min = latest_min - prev_min
        
        metric_html = textwrap.dedent(f"""
            <div class="metric-card">
                <div class="metric-label">Current Min Temp</div>
                <div class="metric-value">{latest_min:.1f}°C</div>
                <div class="metric-delta {'delta-positive' if delta_min >= 0 else 'delta-negative'}">
                    {'↑' if delta_min >= 0 else '↓'} {abs(delta_min):.1f}°C
                </div>
            </div>
            """)
        st.markdown(metric_html, unsafe_allow_html=True)
    
    with col3:
        avg_temp = df["maxtp"].mean()
        metric_html = textwrap.dedent(f"""
            <div class="metric-card">
                <div class="metric-label">Average Max Temp</div>
                <div class="metric-value">{avg_temp:.1f}°C</div>
                <div style="font-size: 0.85rem; opacity: 0.7; margin-top: 0.5rem;">Historical Mean</div>
            </div>
            """)
        st.markdown(metric_html, unsafe_allow_html=True)
    
    with col4:
        total_records = len(df)
        metric_html = textwrap.dedent(f"""
            <div class="metric-card">
                <div class="metric-label">Total Records</div>
                <div class="metric-value">{total_records:,}</div>
                <div style="font-size: 0.85rem; opacity: 0.7; margin-top: 0.5rem;">Data Points</div>
            </div>
            """)
        st.markdown(metric_html, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Temperature trend visualization
    st.markdown('<div class="section-header">📈 Temperature Trends</div>', unsafe_allow_html=True)
    
    if dt_col:
        # Create interactive plot with both max and min temps
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Maximum Temperature", "Minimum Temperature"),
            vertical_spacing=0.12
        )
        
        fig.add_trace(
            go.Scatter(
                x=df[dt_col], y=df["maxtp"],
                name="Max Temp",
                line=dict(color="#2E86DE", width=2),
                fill='tozeroy',
                fillcolor='rgba(46, 134, 222, 0.1)'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df[dt_col], y=df["mintp"],
                name="Min Temp",
                line=dict(color="#5F27CD", width=2),
                fill='tozeroy',
                fillcolor='rgba(95, 39, 205, 0.1)'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            template=theme,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No datetime column for trend visualization.")
    
    # Statistics grid
    st.markdown('<div class="section-header">📊 Statistical Summary</div>', unsafe_allow_html=True)
    
    max_stats = calculate_statistics(df["maxtp"].dropna())
    min_stats = calculate_statistics(df["mintp"].dropna())
    
    col1, col2 = st.columns(2)
    
    with col1:
        info_html = textwrap.dedent("""
            <div class="info-card">
                <h3>Maximum Temperature Statistics</h3>
            </div>
            """)
        st.markdown(info_html, unsafe_allow_html=True)
        
        stats_items = []
        for label, value in [
            ("Mean", max_stats["mean"]),
            ("Std Dev", max_stats["std"]),
            ("Minimum", max_stats["min"]),
            ("Maximum", max_stats["max"]),
            ("Median", max_stats["median"]),
            ("IQR", max_stats["q75"] - max_stats["q25"])
        ]:
            item_html = textwrap.dedent(f"""
                <div class="stat-item">
                    <div class="stat-value">{value:.2f}</div>
                    <div class="stat-label">{label}</div>
                </div>
                """)
            stats_items.append(item_html)
        
        stats_html = '<div class="stats-grid">' + ''.join(stats_items) + '</div>'
        st.markdown(stats_html, unsafe_allow_html=True)
    
    with col2:
        info_html = textwrap.dedent("""
            <div class="info-card">
                <h3>Minimum Temperature Statistics</h3>
            </div>
            """)
        st.markdown(info_html, unsafe_allow_html=True)
        
        stats_items = []
        for label, value in [
            ("Mean", min_stats["mean"]),
            ("Std Dev", min_stats["std"]),
            ("Minimum", min_stats["min"]),
            ("Maximum", min_stats["max"]),
            ("Median", min_stats["median"]),
            ("IQR", min_stats["q75"] - min_stats["q25"])
        ]:
            item_html = textwrap.dedent(f"""
                <div class="stat-item">
                    <div class="stat-value">{value:.2f}</div>
                    <div class="stat-label">{label}</div>
                </div>
                """)
            stats_items.append(item_html)
        
        stats_html = '<div class="stats-grid">' + ''.join(stats_items) + '</div>'
        st.markdown(stats_html, unsafe_allow_html=True)
    
    # Model Performance Summary (if available)
    if tp and hasattr(tp, "predictions"):
        st.markdown('<div class="section-header">🎯 Model Performance</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="model-summary">', unsafe_allow_html=True)
            st.markdown("#### Maximum Temperature Model")
            
            if "maxtp" in tp.predictions:
                from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                
                actual = tp.predictions["maxtp"]["actual"]
                predicted = tp.predictions["maxtp"]["predicted"]
                
                r2 = r2_score(actual, predicted)
                mae = mean_absolute_error(actual, predicted)
                rmse = np.sqrt(mean_squared_error(actual, predicted))
                
                st.metric("R² Score", f"{r2:.4f}")
                st.metric("MAE", f"{mae:.3f}°C")
                st.metric("RMSE", f"{rmse:.3f}°C")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="model-summary">', unsafe_allow_html=True)
            st.markdown("#### Minimum Temperature Model")
            
            if "mintp" in tp.predictions:
                from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                
                actual = tp.predictions["mintp"]["actual"]
                predicted = tp.predictions["mintp"]["predicted"]
                
                r2 = r2_score(actual, predicted)
                mae = mean_absolute_error(actual, predicted)
                rmse = np.sqrt(mean_squared_error(actual, predicted))
                
                st.metric("R² Score", f"{r2:.4f}")
                st.metric("MAE", f"{mae:.3f}°C")
                st.metric("RMSE", f"{rmse:.3f}°C")
            
            st.markdown('</div>', unsafe_allow_html=True)

elif page == "📈 Data Analysis":
    header_html = textwrap.dedent("""
        <div class="main-header">
            <h1>📈 Data Analysis</h1>
            <p>Exploratory Data Analysis & Visualizations</p>
        </div>
        """)
    st.markdown(header_html, unsafe_allow_html=True)
    
    # Data overview
    st.markdown('<div class="section-header">📋 Dataset Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Features", len(df.columns))
    with col3:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100) if len(df) > 0 else 0
        st.metric("Missing Data", f"{missing_pct:.2f}%")
    
    # Data preview
    st.markdown("#### 📊 Data Preview")
    st.dataframe(df.head(20), use_container_width=True)
    
    # Correlation analysis
    st.markdown('<div class="section-header">🔗 Correlation Analysis</div>', unsafe_allow_html=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Feature Correlation Matrix",
            height=600,
            template=theme
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Insufficient numeric columns for correlation analysis.")
    
    # Distribution analysis
    st.markdown('<div class="section-header">📊 Temperature Distributions</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df["maxtp"].dropna(),
            name="Max Temp",
            nbinsx=50,
            marker_color='#2E86DE',
            opacity=0.7
        ))
        fig.update_layout(
            title="Maximum Temperature Distribution",
            xaxis_title="Temperature (°C)",
            yaxis_title="Frequency",
            template=theme,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df["mintp"].dropna(),
            name="Min Temp",
            nbinsx=50,
            marker_color='#5F27CD',
            opacity=0.7
        ))
        fig.update_layout(
            title="Minimum Temperature Distribution",
            xaxis_title="Temperature (°C)",
            yaxis_title="Frequency",
            template=theme,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Box plots
    st.markdown('<div class="section-header">📦 Temperature Ranges</div>', unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(go.Box(y=df["maxtp"].dropna(), name="Max Temp", marker_color='#2E86DE'))
    fig.add_trace(go.Box(y=df["mintp"].dropna(), name="Min Temp", marker_color='#5F27CD'))
    
    fig.update_layout(
        title="Temperature Box Plots",
        yaxis_title="Temperature (°C)",
        template=theme,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif page == "🔬 Model Diagnostics":
    header_html = textwrap.dedent("""
        <div class="main-header">
            <h1>🔬 Model Diagnostics</h1>
            <p>Model Performance Analysis & Validation</p>
        </div>
        """)
    st.markdown(header_html, unsafe_allow_html=True)
    
    if not tp or not hasattr(tp, "predictions"):
        st.warning("⚠️ No trained models available. Please train models first from the sidebar.")
        st.stop()
    
    target = st.selectbox("Select Target Variable", ["maxtp", "mintp"])
    
    if target not in tp.predictions:
        st.error(f"No predictions available for {target}")
        st.stop()
    
    actual = tp.predictions[target]["actual"]
    predicted = tp.predictions[target]["predicted"]
    
    # Performance metrics
    st.markdown('<div class="section-header">🎯 Performance Metrics</div>', unsafe_allow_html=True)
    
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    
    r2 = r2_score(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100 if np.all(actual != 0) else np.nan
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        metric_html = textwrap.dedent(f"""
            <div class="metric-card">
                <div class="metric-label">R² Score</div>
                <div class="metric-value">{r2:.4f}</div>
            </div>
            """)
        st.markdown(metric_html, unsafe_allow_html=True)
    
    with col2:
        metric_html = textwrap.dedent(f"""
            <div class="metric-card">
                <div class="metric-label">MAE</div>
                <div class="metric-value">{mae:.3f}°C</div>
            </div>
            """)
        st.markdown(metric_html, unsafe_allow_html=True)
    
    with col3:
        metric_html = textwrap.dedent(f"""
            <div class="metric-card">
                <div class="metric-label">RMSE</div>
                <div class="metric-value">{rmse:.3f}°C</div>
            </div>
            """)
        st.markdown(metric_html, unsafe_allow_html=True)
    
    with col4:
        metric_html = textwrap.dedent(f"""
            <div class="metric-card">
                <div class="metric-label">MAPE</div>
                <div class="metric-value">{mape:.2f}%</div>
            </div>
            """)
        st.markdown(metric_html, unsafe_allow_html=True)
    
    # Actual vs Predicted
    st.markdown('<div class="section-header">📊 Actual vs Predicted</div>', unsafe_allow_html=True)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Time Series Comparison", "Scatter Plot"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Time series
    fig.add_trace(
        go.Scatter(x=actual.index, y=actual.values, name="Actual", line=dict(color='#2E86DE', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=predicted.index, y=predicted.values, name="Predicted", line=dict(color='#5F27CD', width=2, dash='dash')),
        row=1, col=1
    )
    
    # Scatter plot
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    
    fig.add_trace(
        go.Scatter(x=actual.values, y=predicted.values, mode='markers',
                   name="Predictions", marker=dict(color='#00D2D3', size=6, opacity=0.6)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines',
                   name="Perfect Fit", line=dict(color='#EE5A6F', width=2, dash='dash')),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_xaxes(title_text="Actual (°C)", row=1, col=2)
    fig.update_yaxes(title_text="Predicted (°C)", row=1, col=2)
    
    fig.update_layout(height=500, template=theme, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Residual analysis
    st.markdown('<div class="section-header">📉 Residual Analysis</div>', unsafe_allow_html=True)
    
    residuals = actual - predicted
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Residuals Over Time", "Residual Distribution",
                        "Residuals vs Predicted", "Q-Q Plot")
    )
    
    # Residuals over time
    fig.add_trace(
        go.Scatter(x=residuals.index, y=residuals.values, mode='lines',
                   line=dict(color='#2E86DE', width=1)),
        row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    
    # Histogram
    fig.add_trace(
        go.Histogram(x=residuals.values, nbinsx=40, marker_color='#5F27CD'),
        row=1, col=2
    )
    
    # Residuals vs predicted
    fig.add_trace(
        go.Scatter(x=predicted.values, y=residuals.values, mode='markers',
                   marker=dict(color='#00D2D3', size=4, opacity=0.5)),
        row=2, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
    
    # Q-Q plot
    from scipy import stats
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    fig.add_trace(
        go.Scatter(x=osm, y=osr, mode='markers',
                   marker=dict(color='#FFA801', size=4)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=osm, y=slope*osm + intercept, mode='lines',
                   line=dict(color='red', dash='dash')),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Residual", row=1, col=2)
    fig.update_xaxes(title_text="Predicted", row=2, col=1)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
    
    fig.update_yaxes(title_text="Residual", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    fig.update_yaxes(title_text="Residual", row=2, col=1)
    fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2)
    
    fig.update_layout(height=800, template=theme, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Model summaries
    st.markdown('<div class="section-header">📋 Model Summaries</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["OLS Model", "ARIMA Model", "GARCH Model"])
    
    with tab1:
        if hasattr(tp, "ols_models") and target in tp.ols_models:
            st.markdown(f"### OLS Regression Summary - {target.upper()}")
            summary_text = tp.ols_models[target].summary().as_text()
            st.text(summary_text)
        else:
            st.warning("OLS model summary not available")
    
    with tab2:
        if hasattr(tp, "arima_models") and target in tp.arima_models:
            st.markdown(f"### ARIMA Model Summary - {target.upper()}")
            try:
                summary_text = tp.arima_models[target].summary().as_text()
                st.text(summary_text)
            except:
                st.info("ARIMA summary not available")
        else:
            st.warning("ARIMA model not available")
    
    with tab3:
        if hasattr(tp, "garch_models") and target in tp.garch_models:
            st.markdown(f"### GARCH Model Summary - {target.upper()}")
            try:
                garch_model = tp.garch_models[target]
                summary_text = garch_model.summary().as_text()
                st.text(summary_text)
            except:
                st.info("GARCH summary not available")
        else:
            st.warning("GARCH model not available")

elif page == "🔮 Forecasting":
    header_html = textwrap.dedent("""
        <div class="main-header">
            <h1>🔮 Temperature Forecasting</h1>
            <p>Generate Future Temperature Predictions</p>
        </div>
        """)
    st.markdown(header_html, unsafe_allow_html=True)
    
    if not tp:
        st.warning("⚠️ No trained models available. Please train models first from the sidebar.")
        st.stop()
    
    st.markdown('<div class="section-header">📝 Input Parameters</div>', unsafe_allow_html=True)
    
    with st.form("forecast_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Weather Conditions")
            sun = st.number_input("☀️ Sunshine (hours)",
                                  value=float(df["sun"].dropna().iloc[-1]) if "sun" in df.columns else 5.0,
                                  min_value=0.0, max_value=24.0, step=0.1)
            soil = st.number_input("🌱 Soil Temperature (°C)",
                                   value=float(df["soil"].dropna().iloc[-1]) if "soil" in df.columns else 10.0,
                                   step=0.1)
            rain = st.number_input("🌧️ Rainfall (mm)",
                                   value=float(df["rain"].dropna().iloc[-1]) if "rain" in df.columns else 0.0,
                                   min_value=0.0, step=0.1)
        
        with col2:
            st.markdown("#### Temperature History")
            maxtp_y = st.number_input("🌡️ Yesterday's Max (°C)",
                                      value=float(df["maxtp"].dropna().iloc[-1]) if not df["maxtp"].dropna().empty else 15.0,
                                      step=0.1)
            mintp_y = st.number_input("❄️ Yesterday's Min (°C)",
                                      value=float(df["mintp"].dropna().iloc[-1]) if not df["mintp"].dropna().empty else 5.0,
                                      step=0.1)
        
        with col3:
            st.markdown("#### Other Factors")
            gmin = st.number_input("🌡️ Ground Min Temp (°C)",
                                   value=float(df["gmin"].dropna().iloc[-1]) if "gmin" in df.columns else 5.0,
                                   step=0.1)
            wdsp = st.number_input("💨 Wind Speed (m/s)",
                                   value=float(df["wdsp"].dropna().iloc[-1]) if "wdsp" in df.columns else 10.0,
                                   min_value=0.0, step=0.1)
        
        st.markdown("#### Forecast Date")
        forecast_date = st.date_input("📅 Select Date",
                                      value=datetime.now().date() + timedelta(days=1))
        
        submitted = st.form_submit_button("🚀 Generate Forecast", use_container_width=True)
    
    if submitted:
        params = {
            "sun": sun,
            "soil": soil,
            "rain": rain,
            "gmin": gmin,
            "wdsp": wdsp,
            "maxtp_yesterday": maxtp_y,
            "mintp_yesterday": mintp_y,
            "forecast_date": pd.to_datetime(forecast_date)
        }
        
        try:
            predictions = tp.predict_from_dict(params)
            
            st.markdown('<div class="section-header">🎯 Forecast Results</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                maxtp_hybrid = predictions.get("maxtp", {}).get("Hybrid", 0)
                icon = weather_icon_for_temp(maxtp_hybrid, rain)
                
                forecast_html = textwrap.dedent(f"""
                    <div class="metric-card" style="text-align: center;">
                        <div class="forecast-icon">{icon}</div>
                        <div class="metric-label">Predicted Max Temperature</div>
                        <div class="metric-value">{maxtp_hybrid:.1f}°C</div>
                        <div style="font-size: 0.85rem; opacity: 0.7; margin-top: 0.5rem;">
                            {forecast_date.strftime('%B %d, %Y')}
                        </div>
                    </div>
                    """)
                st.markdown(forecast_html, unsafe_allow_html=True)
            
            with col2:
                mintp_hybrid = predictions.get("mintp", {}).get("Hybrid", 0)
                
                forecast_html = textwrap.dedent(f"""
                    <div class="metric-card" style="text-align: center;">
                        <div class="forecast-icon">❄️</div>
                        <div class="metric-label">Predicted Min Temperature</div>
                        <div class="metric-value">{mintp_hybrid:.1f}°C</div>
                        <div style="font-size: 0.85rem; opacity: 0.7; margin-top: 0.5rem;">
                            {forecast_date.strftime('%B %d, %Y')}
                        </div>
                    </div>
                    """)
                st.markdown(forecast_html, unsafe_allow_html=True)
            
            with col3:
                temp_range = maxtp_hybrid - mintp_hybrid
                
                forecast_html = textwrap.dedent(f"""
                    <div class="metric-card" style="text-align: center;">
                        <div class="forecast-icon">📊</div>
                        <div class="metric-label">Temperature Range</div>
                        <div class="metric-value">{temp_range:.1f}°C</div>
                        <div style="font-size: 0.85rem; opacity: 0.7; margin-top: 0.5rem;">
                            Daily Variation
                        </div>
                    </div>
                    """)
                st.markdown(forecast_html, unsafe_allow_html=True)
            
        #     # Model comparison
        #     st.markdown('<div class="section-header">🔍 Model Comparison</div>', unsafe_allow_html=True)
            
        #     comparison_data = []
        #     for temp_type in ["maxtp", "mintp"]:
        #         pred = predictions.get(temp_type, {})
        #         comparison_data.append({
        #             "Temperature": temp_type.upper(),
        #             "OLS": f"{pred.get('OLS', 0):.2f}°C" if pred.get('OLS') is not None else "N/A",
        #             "ARIMA": f"{pred.get('ARIMA', 0):.2f}°C" if pred.get('ARIMA') is not None else "N/A",
        #             "Hybrid": f"{pred.get('Hybrid', 0):.2f}°C"
        #         })
            
        #     comparison_df = pd.DataFrame(comparison_data)
        #     st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
        # except Exception as e:
        #     st.error(f"Forecast generation failed: {str(e)}")
        #     traceback.print_exc()

elif page == "📥 Export & Reports":
    header_html = textwrap.dedent("""
        <div class="main-header">
            <h1>📥 Export & Reports</h1>
            <p>Download Data, Models & Generate Reports</p>
        </div>
        """)
    st.markdown(header_html, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">📊 Data Export</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📄 Raw Data")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Data CSV",
            data=csv,
            file_name=f"temperature_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        st.markdown("### 📈 Statistics Summary")
        summary_df = df.describe()
        summary_csv = summary_df.to_csv().encode('utf-8')
        st.download_button(
            label="⬇️ Download Statistics CSV",
            data=summary_csv,
            file_name=f"statistics_summary_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    if tp and hasattr(tp, "predictions"):
        st.markdown('<div class="section-header">🤖 Model Exports</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔮 Predictions")
            pred_data = []
            for target in ["maxtp", "mintp"]:
                if target in tp.predictions:
                    actual = tp.predictions[target]["actual"]
                    predicted = tp.predictions[target]["predicted"]
                    for idx in actual.index:
                        pred_data.append({
                            "date": idx,
                            f"{target}_actual": actual[idx],
                            f"{target}_predicted": predicted[idx]
                        })
            
            if pred_data:
                pred_df = pd.DataFrame(pred_data)
                pred_csv = pred_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Predictions CSV",
                    data=pred_csv,
                    file_name=f"model_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            st.markdown("### 💾 Model Files")
            try:
                with open("models.pkl", "rb") as f:
                    model_data = f.read()
                st.download_button(
                    label="⬇️ Download Model (PKL)",
                    data=model_data,
                    file_name=f"trained_models_{datetime.now().strftime('%Y%m%d')}.pkl",
                    mime="application/octet-stream",
                    use_container_width=True
                )
            except:
                st.info("Model file not available")
    
    # Generate report
    st.markdown('<div class="section-header">📋 Generate Report</div>', unsafe_allow_html=True)
    
    if st.button("📄 Generate Full Analysis Report", use_container_width=True):
        report_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        date_range = f"{df[dt_col].min()} to {df[dt_col].max()}" if dt_col else 'N/A'
        report = f"""
# Temperature Forecast Analysis Report
Generated: {report_date}
## Dataset Overview
- Total Records: {len(df):,}
- Features: {len(df.columns)}
- Date Range: {date_range}
## Temperature Statistics
### Maximum Temperature
- Mean: {df['maxtp'].mean():.2f}°C
- Std Dev: {df['maxtp'].std():.2f}°C
- Min: {df['maxtp'].min():.2f}°C
- Max: {df['maxtp'].max():.2f}°C
- Median: {df['maxtp'].median():.2f}°C
### Minimum Temperature
- Mean: {df['mintp'].mean():.2f}°C
- Std Dev: {df['mintp'].std():.2f}°C
- Min: {df['mintp'].min():.2f}°C
- Max: {df['mintp'].max():.2f}°C
- Median: {df['mintp'].median():.2f}°C
"""
        
        if tp and hasattr(tp, "predictions"):
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
            
            report += "\n## Model Performance\n"
            
            for target in ["maxtp", "mintp"]:
                if target in tp.predictions:
                    actual = tp.predictions[target]["actual"]
                    predicted = tp.predictions[target]["predicted"]
                    
                    r2 = r2_score(actual, predicted)
                    mae = mean_absolute_error(actual, predicted)
                    rmse = np.sqrt(mean_squared_error(actual, predicted))
                    
                    report += f"""
### {target.upper()} Model
- R² Score: {r2:.4f}
- MAE: {mae:.3f}°C
- RMSE: {rmse:.3f}°C
"""
        
        st.download_button(
            label="⬇️ Download Report (Markdown)",
            data=report.encode('utf-8'),
            file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown",
            use_container_width=True
        )

# Footer
st.markdown("---")
footer_html = textwrap.dedent("""
    <div style='text-align: center; opacity: 0.7; padding: 2rem 0;'>
        <p style='margin: 0;'>🌡️ Temperature Forecast Dashboard | Advanced Time Series Analysis</p>
        <p style='margin: 0.5rem 0 0 0; font-size: 0.85rem;'>Built with Streamlit & Python</p>
    </div>
    """)
st.markdown(footer_html, unsafe_allow_html=True)

