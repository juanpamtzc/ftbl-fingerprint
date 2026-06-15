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

# --- 2. UI HEADER & EDUCATIONAL GLOSSARY ---
st.title("⚽ Intelligent Tactical Scouting Engine")

with st.expander("📖 How to use this engine (Glossary)"):
    st.markdown("""
    **For Non-Technical Users:**
    * **Self-Variance Multiplier:** Players are not robots; they play slightly differently every game. A score of `1.0x` means a replacement plays exactly as similar to the target as the target does to themselves. 
    * **Scores < 1.0x (True Twins):** The replacement is indistinguishable from the target.
    * **Scores > 2.0x (Different Profile):** The replacement plays a fundamentally different role on the pitch.
    
    **For Technical Users:**
    * **PCA (Linear Baseline):** Projects 112 features onto 16 orthogonal axes. Best for isolating massive volume differences (like total passes or total distance).
    * **VAE (Deep Neural Network):** A Variational Autoencoder that folds the 112 dimensions into a non-linear continuous space. Best for capturing complex, contextual behaviors.
    """)

st.divider()

# --- 3. SEARCH CONTROLS ---
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 1])
with col_ctrl1:
    target_player = st.selectbox("Target Player to Scout:", players, 
                                 help="Select the player you are trying to replace.")
with col_ctrl2:
    model_choice = st.radio("Underlying AI Architecture:", 
                            ["PCA (Linear Baseline)", "VAE (Non-Linear Manifold)"], 
                            horizontal=True,
                            help="Toggle the mathematical brain used to calculate similarities.")
with col_ctrl3:
    n_results = st.slider("Top N Replacements:", 1, 10, 5, 
                          help="Number of similar players to display in the table.")

# --- 4. DISTANCE CALCULATION & NORMALIZATION ---
active_df = pca_centroids if "PCA" in model_choice else vae_centroids
baseline_dist = metadata.loc["PCA" if "PCA" in model_choice else "VAE", "Self_Distance_Baseline"]

target_vector = active_df.loc[target_player].values.reshape(1, -1)
distances = cdist(target_vector, active_df.values, metric='euclidean').flatten()

norm_distances = distances / baseline_dist

df_results = pd.DataFrame({
    'Player': active_df.index,
    'Distance Multiplier': norm_distances
}).sort_values('Distance Multiplier')

# Filter out the target player themselves
df_others = df_results[df_results['Player'] != target_player]
df_replacements = df_others.head(n_results)
best_match = df_replacements.iloc[0]['Player']

# --- 5. THE REPLACEABILITY BANNER ---
clones_count = len(df_others[df_others['Distance Multiplier'] <= 1.0])
peers_count = len(df_others[(df_others['Distance Multiplier'] > 1.0) & (df_others['Distance Multiplier'] <= 2.0)])

if clones_count == 0:
    replaceability_tier = "🦄 IRREPLACEABLE (Unicorn)"
    banner_color = "normal" # Default Streamlit blue/grey
    st.info(f"**{target_player} is a Unicorn.** There are no players in this database who replicate their output seamlessly. You will have to change your tactical system if they leave.", icon="⚠️")
elif clones_count <= 3:
    replaceability_tier = "⭐ HARD TO REPLACE (Rare)"
    st.warning(f"**{target_player} has a Rare Profile.** There are only {clones_count} true tactical twins in the league. Securing a replacement will be highly competitive.", icon="🔍")
else:
    replaceability_tier = "🔄 REPLACEABLE (Standard Profile)"
    st.success(f"**{target_player} has a Standard Profile.** There are {clones_count} players who provide an identical tactical footprint. You have high leverage in the transfer market.", icon="✅")

col_metric1, col_metric2, col_metric3 = st.columns(3)
col_metric1.metric("Market Status", replaceability_tier)
col_metric2.metric("True Twins (< 1.0x Variance)", f"{clones_count} Players", help="Players statistically indistinguishable from the target.")
col_metric3.metric("Tactical Peers (1.0x - 2.0x Variance)", f"{peers_count} Players", help="Players who play similarly but have distinct individual quirks.")

st.divider()

# --- 6. TOP PANELS: TABLE & POPULATION PLOT ---
col_table, col_plot = st.columns([1, 1.5])

with col_table:
    st.subheader(f"Closest Statistical Matches")
    st.dataframe(
        df_replacements.style.background_gradient(
            cmap='RdYlGn_r', vmin=0.5, vmax=5.0, subset=['Distance Multiplier']
        ).format({'Distance Multiplier': "{:.2f}x"}),
        use_container_width=True, hide_index=True
    )

with col_plot:
    st.subheader("League Similarity Distribution")
    fig = px.histogram(
        df_others, 
        x="Distance Multiplier", 
        nbins=40,
        color_discrete_sequence=['#4C72B0'],
        labels={'Distance Multiplier': 'Similarity Score (Lower is better)'}
    )
    fig.add_vline(x=1.0, line_dash="dash", line_color="green", annotation_text="Target's Own Variance (1.0x)")
    fig.add_vline(x=2.0, line_dash="dot", line_color="orange", annotation_text="Tactical Peer Limit (2.0x)")
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=300)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- 7. BOTTOM PANELS: HEAD-TO-HEAD COMPARISON ---
st.subheader(f"Head-to-Head Proof: {target_player} vs {best_match}")

col_pitch, col_stats = st.columns([1, 1.2])

with col_pitch:
    st.markdown("**Spatial Heatmap (Where they play)**")
    
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
    st.markdown("**Tactical & Kinematic Profiles (How they play)**")
    
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