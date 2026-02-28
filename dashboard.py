import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="Urban Environmental Monitoring", layout="wide")

st.title("🌍 Environmental Air Quality Monitoring Dashboard")

# Quick help in sidebar
with st.sidebar:
    st.markdown("### How to use")
    st.caption("Select a zone and pollutant on the left, then explore the analysis tabs. All charts update in real-time.")

from utils import load_openaq

# -----------------------------
# LOAD & PREPROCESS RAW OPENAQ DATA (cached)
# -----------------------------
@st.cache_data
def load_data(raw_path: str = "data/openaq_2025.csv"):
    return load_openaq(raw_path)

# load dataset once
df = load_data()

# sidebar: Zone and Variable selection (always visible and simple)
st.sidebar.markdown("---")
st.sidebar.header("📊 Filters")
zone = st.sidebar.radio("Zone", ["All", "Industrial", "Residential"], index=0, horizontal=False)
features = ["pm25", "pm10", "no2", "ozone", "temperature", "humidity"]
pollutant = st.sidebar.selectbox("Pollutant/Variable", features)

# Date range selector
min_date = df["datetime"].min().date()
max_date = df["datetime"].max().date()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Apply filters
if zone != "All":
    df = df[df["zone"] == zone]

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df = df[(df.datetime >= start_date) & (df.datetime <= end_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))]

# Top-level KPIs
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
col1.metric(f"📈 Avg {pollutant.upper()}", f"{df[pollutant].mean():.1f}")
col2.metric(f"🔴 Max {pollutant.upper()}", f"{df[pollutant].max():.1f}")
temp_avg = df['temperature'].mean()
col3.metric(f"🌡️ Avg Temperature", f"{temp_avg:.1f}°C" if not pd.isna(temp_avg) else "N/A")
col4.metric(f"📊 Total Records", f"{len(df):,}")

st.markdown("---")

# Tab-based interface for each analysis
tabs = st.tabs(["📊 PCA Analysis", "📈 Time Series", "📉 Distribution", "✅ Audit"])

# ============================================================================
# TAB 1: PCA (Dimensionality Reduction)
# ============================================================================
with tabs[0]:
    st.subheader("Principal Component Analysis")
    st.caption("Identify dominant patterns across 6 variables using PCA.")
    st.info("🔄 *Zone filter is ignored for PCA so you can always compare Industrial vs Residential clusters.*")
    
    # PCA should always include both zones; ignore the sidebar zone filter.
    pca_source = load_data()
    # still respect the selected date range for comparability
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        pca_source = pca_source[(pca_source.datetime >= start_date) & (pca_source.datetime <= end_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))]

    features_list = ["pm25", "pm10", "no2", "ozone", "temperature", "humidity"]
    # Prepare matrix (drop rows where all features are NaN)
    X = pca_source[features_list].dropna(how='all')
    X = X.ffill().bfill()
    X = X.dropna()

    if X.shape[0] < 2:
        st.warning("⚠️ Insufficient data for PCA. Try selecting a wider date range.")
    else:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(Xs)

        pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
        # use zone from the unfiltered source so both appear
        pca_df["zone"] = pca_source.loc[X.index, "zone"].values

        # Create two-column layout
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            fig_pca = px.scatter(
                pca_df.sample(min(len(pca_df), 10000)),
                x="PC1",
                y="PC2",
                color="zone",
                opacity=0.7,
                title="PCA Projection (2 Components)",
                labels={"PC1": f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", 
                       "PC2": f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"}
            )
            fig_pca.update_layout(height=500)
            st.plotly_chart(fig_pca, use_container_width=True)
        
        with col2:
            loadings = pd.DataFrame(
                pca.components_.T,
                index=features_list,
                columns=["PC1", "PC2"]
            )
            abs_load = loadings.abs().sum(axis=1).sort_values(ascending=True).reset_index()
            abs_load.columns = ["feature", "contribution"]
            
            fig_load = px.bar(
                abs_load,
                x="contribution",
                y="feature",
                orientation="h",
                title="Variable Contributions",
                color="contribution",
                color_continuous_scale="Blues"
            )
            fig_load.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig_load, use_container_width=True)

# ============================================================================
# TAB 2: Time Series (Temporal Analysis)
# ============================================================================
with tabs[1]:
    st.subheader("Time Series & Temporal Patterns")
    st.caption("Monitor pollutant levels over time and identify daily/seasonal patterns.")
    
    # Aggregate hourly data for smoother visualization
    ts_data = df.groupby(pd.Grouper(key="datetime", freq="h"))[pollutant].agg(["mean", "count"]).reset_index()
    ts_data = ts_data[ts_data["count"] > 0]  # Remove empty hours
    ts_data.columns = ["datetime", "value", "count"]
    
    if ts_data.empty:
        st.warning("No data for the selected filters.")
    else:
        # Add 24-hour rolling average
        ts_data["trend"] = ts_data["value"].rolling(window=24, center=True).mean()
        
        # Create interactive time series with trend
        fig_ts = go.Figure()
        
        # Add actual values as light line
        fig_ts.add_trace(go.Scatter(
            x=ts_data["datetime"],
            y=ts_data["value"],
            name="Actual (Hourly)",
            mode="lines",
            line=dict(color="lightsteelblue", width=1),
            opacity=0.6
        ))
        
        # Add trend line
        fig_ts.add_trace(go.Scatter(
            x=ts_data["datetime"],
            y=ts_data["trend"],
            name="24-Hour Trend",
            mode="lines",
            line=dict(color="darkblue", width=3),
            opacity=0.9
        ))
        
        date_start = ts_data["datetime"].min().strftime("%Y-%m-%d")
        date_end = ts_data["datetime"].max().strftime("%Y-%m-%d")
        
        fig_ts.update_layout(
            title=f"<b>{pollutant.upper()} - Hourly Monitoring ({date_start} to {date_end})</b>",
            xaxis_title="Date & Time",
            yaxis_title=f"{pollutant} Level",
            height=500,
            hovermode="x unified",
            template="plotly_white"
        )
        st.plotly_chart(fig_ts, use_container_width=True)
        
        # Show summary stats for the time series
        st.markdown("**Time Series Summary**")
        ts_summary = pd.DataFrame({
            "Metric": ["Min Value", "Max Value", "Mean", "Std Dev", "Data Points"],
            "Value": [
                f"{ts_data['value'].min():.2f}",
                f"{ts_data['value'].max():.2f}",
                f"{ts_data['value'].mean():.2f}",
                f"{ts_data['value'].std():.2f}",
                f"{len(ts_data)}"
            ]
        })
        st.dataframe(ts_summary, use_container_width=True, hide_index=True)
        
        # Daily heatmap: show patterns day-by-day
        heat_data = df.copy()
        heat_data["day_of_year"] = heat_data["datetime"].dt.dayofyear
        heat_data["hour"] = heat_data["datetime"].dt.hour
        
        pivot = heat_data.pivot_table(
            index="hour",
            columns="day_of_year",
            values=pollutant,
            aggfunc="mean"
        )
        
        if not pivot.empty and pivot.shape[1] > 1:
            fig_hm = px.imshow(
                pivot,
                aspect="auto",
                color_continuous_scale="Reds",
                title=f"{pollutant.upper()} - Daily Pattern Heatmap (Hour × Day)",
                labels={"x": "Day of Year", "y": "Hour of Day", "color": pollutant}
            )
            fig_hm.update_layout(height=400)
            st.plotly_chart(fig_hm, use_container_width=True)

# ============================================================================
# TAB 3: Distribution Analysis
# ============================================================================
with tabs[2]:
    st.subheader("Distribution Analysis")
    st.caption("Understand the statistical distribution and identify outliers.")
    
    poll_vals = df[pollutant].dropna()
    
    if len(poll_vals) == 0:
        st.warning("No data for the selected filters.")
    else:
        # Calculate key percentiles
        p25 = np.percentile(poll_vals, 25)
        p50 = np.percentile(poll_vals, 50)
        p75 = np.percentile(poll_vals, 75)
        p95 = np.percentile(poll_vals, 95)
        p99 = np.percentile(poll_vals, 99)
        
        # Display KPIs
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Q1 (25th %ile)", f"{p25:.1f}")
        col2.metric("Median", f"{p50:.1f}")
        col3.metric("Q3 (75th %ile)", f"{p75:.1f}")
        col4.metric("95th %ile", f"{p95:.2f}")
        col5.metric("99th %ile", f"{p99:.2f}")
        
        st.markdown("---")
        
        # Main distribution plot
        fig_dist = px.histogram(
            poll_vals,
            nbins=60,
            title=f"{pollutant.upper()} Distribution",
            labels={pollutant: "Value", "count": "Frequency"},
            color_discrete_sequence=["steelblue"]
        )
        
        # Add percentile markers (without overlapping text)
        fig_dist.add_vline(p50, line_dash="dash", line_color="orange", name="Median", annotation_text="Median", annotation_position="top right")
        fig_dist.add_vline(p95, line_dash="dash", line_color="red", name="95th %ile")
        fig_dist.update_layout(margin=dict(t=100), hovermode="closest")
        
        fig_dist.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Summary stats
        st.markdown("**Summary Statistics**")
        summary = pd.DataFrame({
            "Statistic": ["Count", "Mean", "Std Dev", "Min", "Max"],
            "Value": [
                f"{len(poll_vals)}",
                f"{poll_vals.mean():.2f}",
                f"{poll_vals.std():.2f}",
                f"{poll_vals.min():.2f}",
                f"{poll_vals.max():.2f}"
            ]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

# ============================================================================
# TAB 4: Visual Audit
# ============================================================================
with tabs[3]:
    st.subheader("Zone Comparison & Data Quality")
    st.caption("Compare average levels across zones and download data for further analysis.")
    
    # Reload original unfiltered data for zone comparison
    df_full = load_data()
    
    # Zone-level comparison
    zone_stats = df_full.groupby("zone")[pollutant].agg(["mean", "std", "min", "max", "count"]).reset_index()
    zone_stats.columns = ["Zone", "Average", "Std Dev", "Min", "Max", "Samples"]
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        fig_comp = px.bar(
            df_full.groupby("zone")[pollutant].mean().reset_index(),
            x="zone",
            y=pollutant,
            title=f"Average {pollutant.upper()} by Zone",
            labels={"zone": "Zone", pollutant: f"Average {pollutant}"},
            color="zone",
            color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"]
        )
        fig_comp.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_comp, use_container_width=True)
    
    with col2:
        st.markdown("**Zone Statistics**")
        st.dataframe(zone_stats, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Data export
    st.markdown("**Export Filtered Data**")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Current Data (CSV)",
        data=csv,
        file_name=f"environmental_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption("Dashboard is interactive. Use filters on the left to explore different zones, variables, and time periods.")
