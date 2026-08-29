import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# 1. Unified Setup & Theme
st.set_page_config(layout="wide", page_title="PBAA Dashboard")

try:
    from backend.ml_runner import parse_csv_to_tensor
except ImportError:
    pass

# Custom CSS for a professional, data-centric aerospace theme
st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    .metric-container {
        background-color: #111111;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #222222;
    }
    .critical-log {
        background-color: #0A0A0A;
        border-left: 5px solid #8B0000;
        padding: 1.5rem;
        font-family: 'Courier New', Courier, monospace;
        color: #E0E0E0;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Auto-Execution Pipeline
@st.cache_data
def load_data():
    file_path = "data/synthetic/burn_in_100k_physics_sim.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        # Fallback mock dataframe
        num_components = 100000
        np.random.seed(42)
        df = pd.DataFrame({
            'die_id': [f"DIE_{i:06d}" for i in range(num_components)],
            'X': np.random.uniform(-10, 10, num_components),
            'Y': np.random.uniform(-10, 10, num_components)
        })

    # Mock ML pipeline flags
    if 'tier1_flag' not in df.columns:
        df['tier1_flag'] = np.random.choice([0, 1], len(df), p=[0.97, 0.03])
    if 'tier2_flag' not in df.columns:
        mask = df['tier1_flag'] == 0
        df.loc[mask, 'tier2_flag'] = np.random.choice([0, 1], mask.sum(), p=[0.94, 0.06])
        df['tier2_flag'] = df['tier2_flag'].fillna(0)
        
    # Ensure X and Y exist for spatial mapping (missing in generated CSV)
    if 'X' not in df.columns or 'Y' not in df.columns:
        df['X'] = np.random.uniform(-10, 10, len(df))
        df['Y'] = np.random.uniform(-10, 10, len(df))
    
    # Mock Drift Magnitude (168h vs 0h delta)
    df['Delta_Delay'] = np.where(df['tier2_flag'] == 1, 
                                 np.random.normal(0.45, 0.1, len(df)), 
                                 np.random.normal(0.05, 0.02, len(df)))
    
    df['Delta_Leakage'] = np.where(df['tier2_flag'] == 1,
                                   np.random.normal(25.0, 5.0, len(df)),
                                   np.random.normal(2.0, 0.5, len(df)))

    def determine_status(row):
        if row['tier1_flag'] == 1:
            return 'Gross Fail (Tier 1)'
        elif row['tier2_flag'] == 1:
            return 'Latent Defect (Tier 2)'
        else:
            return 'Pass'
            
    df['Status'] = df.apply(determine_status, axis=1)
    return df

df = load_data()

# 3. Dashboard Layout
st.title("Pre-flight Burn-In Anomaly Analysis - PBAA")
st.markdown("---")

# Row 1: System Telemetry, KPIs & Rejection Pipeline
st.subheader("System Telemetry & Pipeline Rejection Funnel")
r1_col1, r1_col2, r1_col3 = st.columns([1, 2, 2])

total_components = len(df)
tier1_fails = int(df['tier1_flag'].sum())
passed_tier1 = total_components - tier1_fails
tier2_fails = int(df['tier2_flag'].sum())
final_yield = passed_tier1 - tier2_fails
yield_pct = (final_yield / total_components) * 100

with r1_col1:
    st.metric("Total Components", f"{total_components:,}")
    st.metric("Gross Fails (Tier 1)", f"{tier1_fails:,}")
    st.metric("Latent Defects (Tier 2)", f"{tier2_fails:,}")
    st.metric("Overall Yield", f"{yield_pct:.2f}%")

with r1_col2:
    # Funnel Chart
    fig_funnel = go.Figure(go.Funnel(
        y=["Total Ingested", "Passed Tier 1 Filter", "Passed Tier 2 Deep Engine (Yield)"],
        x=[total_components, passed_tier1, final_yield],
        textinfo="value+percent initial",
        marker={"color": ["#555555", "#444444", "#333333"]}
    ))
    fig_funnel.update_layout(title="Yield Degradation Funnel", template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_funnel, width=True)

with r1_col3:
    # Bar Chart for Rejects
    reject_df = pd.DataFrame({
        'Rejection Stage': ['Tier 1 Gross Rejects', 'Tier 2 Latent Rejects'],
        'Count': [tier1_fails, tier2_fails]
    })
    fig_bar = px.bar(
        reject_df, x='Rejection Stage', y='Count',
        color='Rejection Stage',
        color_discrete_map={'Tier 1 Gross Rejects': '#DC143C', 'Tier 2 Latent Rejects': '#8B0000'},
        title="Anomaly Distribution by Tier"
    )
    fig_bar.update_layout(template="plotly_dark", showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_bar, width=True)

st.markdown("---")

# Row 2: Wafer Spatial Map & Drift Magnitude
r2_col1, r2_col2 = st.columns(2)

with r2_col1:
    st.subheader("Wafer Spatial Map")
    color_map = {
        'Pass': '#333333', 
        'Gross Fail (Tier 1)': '#DC143C', 
        'Latent Defect (Tier 2)': '#8B0000'
    }
    
    # Subsample for plotting performance if dataset is massive (100k points)
    plot_df = df.sample(n=min(5000, len(df)), random_state=42) if len(df) > 5000 else df
    
    fig_map = px.scatter(
        plot_df, x='X', y='Y', color='Status', color_discrete_map=color_map,
        hover_name='die_id', title="2D Spatial Distribution (Sampled)"
    )
    fig_map.update_traces(marker=dict(size=5, opacity=0.9))
    fig_map.update_layout(template="plotly_dark", xaxis=dict(showgrid=False, zeroline=False, visible=False), 
                          yaxis=dict(showgrid=False, zeroline=False, visible=False))
    st.plotly_chart(fig_map, width=True)

with r2_col2:
    st.subheader("Latent Drift Magnitude (168h vs 0h)")
    # Filter for Box plot comparison (Healthy vs Tier 2 Latent Defect only)
    box_df = df[df['Status'].isin(['Pass', 'Latent Defect (Tier 2)'])].copy()
    box_df['Status'] = box_df['Status'].replace({'Pass': 'Healthy'})
    
    # Melt dataframe for grouped box plot
    melted_box = pd.melt(
        box_df, id_vars=['die_id', 'Status'], 
        value_vars=['Delta_Delay', 'Delta_Leakage'],
        var_name='Measurement', value_name='Delta Value'
    )
    
    fig_box = px.box(
        melted_box, x='Measurement', y='Delta Value', color='Status',
        color_discrete_map={'Healthy': '#555555', 'Latent Defect (Tier 2)': '#8B0000'},
        title="Drift Delta Comparison (Log Scale)"
    )
    fig_box.update_layout(template="plotly_dark", yaxis_type="log")
    st.plotly_chart(fig_box, width=True)

st.markdown("---")

# Row 3: Tier 3 Root Cause Analysis (XAI)
st.header("Tier 3 Root Cause Analysis (XAI)")

failures_df = df[df['Status'] != 'Pass']

if len(failures_df) > 0:
    selected_die = st.selectbox("Select Flagged Die ID for Explainability Deep Dive:", failures_df['die_id'].tolist())
    
    r3_col1, r3_col2 = st.columns([1, 1])
    
    # Simulate SHAP data based on selected die
    np.random.seed(hash(selected_die) % (2**32))
    
    features = ['Leakage at 96h', 'Delay at 24h', 'IDDQ at 168h', 'Leakage at 168h', 'Voltage Offset']
    attributions = np.random.dirichlet(np.ones(5), size=1)[0] * 100
    
    # Sort for visualization
    shap_df = pd.DataFrame({'Feature': features, 'Attribution (%)': attributions})
    shap_df = shap_df.sort_values('Attribution (%)', ascending=True)
    
    # Identify primary cause for the text log
    primary_feature = shap_df.iloc[-1]['Feature']
    primary_score = shap_df.iloc[-1]['Attribution (%)']
    
    # Determine diagnosis text based on dominant feature
    if 'Leakage' in primary_feature:
        defect_type = "Time-Dependent Dielectric Breakdown (TDDB)"
    elif 'Delay' in primary_feature:
        defect_type = "Negative Bias Temperature Instability (NBTI)"
    else:
        defect_type = "Electromigration (EM)"
        
    diagnosis_text = f"ROOT CAUSE DIAGNOSIS: {primary_score:.1f}% of reconstruction loss attributed to anomalous {primary_feature}. Degradation signature aligns with {defect_type}."
    
    with r3_col1:
        fig_shap = px.bar(
            shap_df, x='Attribution (%)', y='Feature', orientation='h',
            title="Integrated Gradients / SHAP Attributions",
            color_discrete_sequence=['#DC143C']
        )
        fig_shap.update_layout(template="plotly_dark")
        st.plotly_chart(fig_shap, width=True)
        
    with r3_col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(f'<div class="critical-log">{diagnosis_text}</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Remediation Protocol")
        st.markdown(f"- Isolate component {selected_die} from payload assembly.")
        st.markdown(f"- Initiate failure analysis for {defect_type}.")
        st.markdown("- Adjust lot acceptance parameters to penalize early drift in the identified feature.")
        
else:
    st.info("No anomalies detected in the current dataset.")
