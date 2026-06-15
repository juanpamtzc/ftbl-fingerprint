import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="AI Scouting Engine", layout="wide")

# --- 1. LOAD PRE-COMPUTED DATA ---
@st.cache_data
def load_data():
    pca_df = pd.read_csv("pca_centroids.csv", index_col="Player")
    vae_df = pd.read_csv("vae_centroids.csv", index_col="Player")
    raw_df = pd.read_csv("raw_feature_centroids.csv", index_col="Player")
    meta_df = pd.read_csv("model_metadata.csv", index_col="Model")
    return pca_df, vae_df, raw_df, meta_df

pca_centroids, vae_centroids, raw_centroids, metadata = load_data()
players = sorted(pca_centroids.index.tolist())

tactical_keys = [
    'passes_attempted', 'pass_completion_pct', 'shots', 'goals', 'assists', 
    'carries', 'dribbles', 'fouls_won', 'interceptions', 'tackles', 'clearances',
    'avg_action_velocity', 'max_action_velocity', 'explosive_bursts', 
    'total_action_distance', 'avg_deceleration'
]

# --- 2. UI HEADER ---
st.title("⚽ Intelligent Latent Scouting Engine")
st.markdown("""
**Distance Metric (Self-Variance Units):** A score of `1.0x` means the replacement plays exactly as similar to the target as the target does to themselves on average. A global color scale is used: green indicates a true tactical twin, while red indicates no similar players exist.
""")

col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 1])
with col_ctrl1:
    target_player = st.selectbox("Target Player:", players)
with col_ctrl2:
    model_choice = st.radio("Underlying Architecture:", ["PCA (Linear Baseline)", "VAE (Non-Linear Manifold)"], horizontal=True)
with col_ctrl3:
    n_results = st.slider("Top N Replacements:", 1, 10, 5)

# --- 3. DISTANCE CALCULATION & NORMALIZATION ---
active_df = pca_centroids if "PCA" in model_choice else vae_centroids
baseline_dist = metadata.loc["PCA" if "PCA" in model_choice else "VAE", "Self_Distance_Baseline"]

target_vector = active_df.loc[target_player].values.reshape(1, -1)
distances = cdist(target_vector, active_df.values, metric='euclidean').flatten()

# Normalize distances
norm_distances = distances / baseline_dist

df_results = pd.DataFrame({
    'Player': active_df.index,
    'Distance (Self-Variance Multiplier)': norm_distances
}).sort_values('Distance (Self-Variance Multiplier)')

df_replacements = df_results[df_results['Player'] != target_player].head(n_results)
best_match = df_replacements.iloc[0]['Player']

# --- 4. TOP PANELS: TABLE & POPULATION PLOT ---
col_table, col_plot = st.columns([1, 1.5])

with col_table:
    st.subheader(f"Closest Tactical Twins")
    # Apply global color scale (0.0 to 5.0)
    st.dataframe(
        df_replacements.style.background_gradient(
            cmap='RdYlGn_r', vmin=0.5, vmax=5.0, subset=['Distance (Self-Variance Multiplier)']
        ).format({'Distance (Self-Variance Multiplier)': "{:.2f}x"}),
        use_container_width=True, hide_index=True
    )

with col_plot:
    st.subheader("Global Player Similarity Distribution")
    # Plotly Histogram of all players
    fig = px.histogram(
        df_results[df_results['Player'] != target_player], 
        x="Distance (Self-Variance Multiplier)", 
        nbins=50,
        color_discrete_sequence=['#4C72B0'],
        labels={'Distance (Self-Variance Multiplier)': 'Similarity (Lower is better)'}
    )
    # Add threshold lines
    fig.add_vline(x=1.0, line_dash="dash", line_color="green", annotation_text="Self-Variance (1.0x)")
    fig.add_vline(x=df_replacements.iloc[-1]['Distance (Self-Variance Multiplier)'], 
                  line_dash="dot", line_color="red", annotation_text=f"Top {n_results} Cutoff")
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=300)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- 5. BOTTOM PANELS: HEAD-TO-HEAD COMPARISON ---
st.subheader(f"Head-to-Head: {target_player} vs {best_match}")

col_pitch, col_stats = st.columns([1, 1.2])

with col_pitch:
    st.markdown("**Spatial Heatmap (Pitch Dominance)**")
    
    target_spatial = raw_centroids.loc[target_player][[f'Spatial_{i}' for i in range(96)]].values.reshape(8, 12)
    match_spatial = raw_centroids.loc[best_match][[f'Spatial_{i}' for i in range(96)]].values.reshape(8, 12)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    axes[0].imshow(target_spatial, cmap='magma', extent=[0, 120, 0, 80], origin='lower')
    axes[0].set_title(f"{target_player}\n(Target)")
    axes[0].axis('off')
    
    axes[1].imshow(match_spatial, cmap='magma', extent=[0, 120, 0, 80], origin='lower')
    axes[1].set_title(f"{best_match}\n(Closest Match)")
    axes[1].axis('off')
    
    st.pyplot(fig)

with col_stats:
    st.markdown("**Tactical & Kinematic Profiles (Scaled)**")
    
    target_stats = raw_centroids.loc[target_player][tactical_keys].values
    match_stats = raw_centroids.loc[best_match][tactical_keys].values
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=target_stats, theta=tactical_keys, fill='toself', name=target_player, marker=dict(color='blue')
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=match_stats, theta=tactical_keys, fill='toself', name=best_match, marker=dict(color='orange')
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=False)),
        showlegend=True, height=400, margin=dict(l=40, r=40, t=20, b=20)
    )
    st.plotly_chart(fig_radar, use_container_width=True)