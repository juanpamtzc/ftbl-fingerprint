"""
Tactical Engine - Streamlit Frontend (Deep Learning Edition)
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from mplsoccer import Pitch
from scipy.spatial.distance import cosine
import torch

# Import our backend functions
from src.data_fetching import get_coupled_signatures
from src.ml_models import train_autoencoder

st.set_page_config(page_title="Deep Tactical Engine", layout="wide")

@st.cache_data
def fetch_player_data(player_name, comp_id, season_id):
    try:
        X_spatial, X_tactical, X_coupled, features = get_coupled_signatures(
            competition_id=comp_id, 
            season_id=season_id, 
            player_name=player_name
        )
        return X_spatial, X_tactical, X_coupled, features
    except Exception as e:
        return None, None, None, None

# --- UI LAYOUT ---
st.title("🧠 Deep Tactical Signature Profiling")
st.markdown("Map non-linear tactical footprints and latent space DNA between players.")

st.sidebar.header("Match Selection Parameters")
comp_id = st.sidebar.number_input("Competition ID (StatsBomb)", value=11)
season_id = st.sidebar.number_input("Season ID", value=22)

st.sidebar.header("Player A")
player_a = st.sidebar.text_input("Full Name A", value="Lionel Andrés Messi Cuccittini")

st.sidebar.header("Player B")
player_b = st.sidebar.text_input("Full Name B", value="Andrés Iniesta Luján")

analyze_btn = st.sidebar.button("Generate Deep Profiles")

if analyze_btn:
    with st.spinner(f"Extracting match vectors for {player_a} and {player_b}..."):
        sp_A, tac_A, coup_A, feat_A = fetch_player_data(player_a, comp_id, season_id)
        sp_B, tac_B, coup_B, feat_B = fetch_player_data(player_b, comp_id, season_id)
        
        if sp_A is None or len(sp_A) == 0:
            st.error(f"Could not find sufficient data for {player_a}.")
            st.stop()
        if sp_B is None or len(sp_B) == 0:
            st.error(f"Could not find sufficient data for {player_b}.")
            st.stop()

    with st.spinner("Training PyTorch Autoencoder on player match histories..."):
        # Combine data to train a shared Autoencoder space
        X_combined = np.vstack((coup_A, coup_B))
        shared_ae = train_autoencoder(X_combined, latent_dim=3, epochs=800)
        
        # Extract Latent DNA
        shared_ae.eval()
        with torch.no_grad():
            _, latent_A = shared_ae(torch.FloatTensor(coup_A))
            _, latent_B = shared_ae(torch.FloatTensor(coup_B))
            
            latent_A = latent_A.numpy()
            latent_B = latent_B.numpy()

    # --- PITCH VISUALIZATION BLOCK ---
    col1, col2 = st.columns(2)
    
    # Clean up the ugly database labels
    clean_labels = {
        'passes_attempted': 'Passes Attempted',
        'pass_completion_pct': 'Pass Completion %',
        'shots': 'Shots',
        'goals': 'Goals',
        'assists': 'Assists',
        'carries': 'Carries',
        'dribbles': 'Dribbles',
        'fouls_won': 'Fouls Won'
    }
    clean_feat_A = [clean_labels.get(f, f) for f in feat_A]
    
    def draw_pitch_heatmap(spatial_vector, title):
        grid = spatial_vector.reshape(16, 24)
        fig, ax = plt.subplots(figsize=(8, 5))
        pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
        pitch.draw(ax=ax)
        ax.imshow(grid, extent=(0, 120, 80, 0), aspect='auto', cmap='magma', alpha=0.7)
        ax.set_title(title, color='white', fontsize=14)
        fig.patch.set_facecolor('#22312b')
        return fig

    with col1:
        st.subheader(f"{player_a.split()[-2] if len(player_a.split()) > 1 else player_a}")
        st.pyplot(draw_pitch_heatmap(np.mean(sp_A, axis=0), "Average Spatial Footprint"))
        
        # Display cleaner bar chart with custom color
        df_A = pd.DataFrame({"Metric": clean_feat_A, "Impact (Std Devs)": np.mean(tac_A, axis=0)})
        st.bar_chart(df_A.set_index("Metric"), color="#00ffcc")

    with col2:
        st.subheader(f"{player_b.split()[-2] if len(player_b.split()) > 1 else player_b}")
        st.pyplot(draw_pitch_heatmap(np.mean(sp_B, axis=0), "Average Spatial Footprint"))
        
        # Display cleaner bar chart with custom color
        df_B = pd.DataFrame({"Metric": clean_feat_A, "Impact (Std Devs)": np.mean(tac_B, axis=0)})
        st.bar_chart(df_B.set_index("Metric"), color="#ff0066")

    # --- LATENT SPACE 3D VISUALIZATION ---
    st.markdown("---")
    st.subheader("🌌 The Tactical Latent Space")
    
    # Add an explicit explanation for the user
    st.info("""
    **How to read this chart:**
    * **Every dot** represents a single match played by the player.
    * **The Axes** represent the core tactical drives discovered by the neural network (e.g., Playmaking, Finishing).
    * **Close together:** If dots are clustered tightly, the player is highly consistent. If dots of two different players overlap, they fulfilled the exact same tactical role on the pitch.
    """)
    
    df_latent_A = pd.DataFrame(latent_A, columns=["Playmaking Drive", "Pressure/Retention", "Finishing/Threat"])
    df_latent_A["Player"] = player_a
    
    df_latent_B = pd.DataFrame(latent_B, columns=["Playmaking Drive", "Pressure/Retention", "Finishing/Threat"])
    df_latent_B["Player"] = player_b
    
    df_latent_combined = pd.concat([df_latent_A, df_latent_B])
    
    # Add Hover Data and Clean Axes
    fig_3d = px.scatter_3d(
        df_latent_combined, 
        x="Playmaking Drive", y="Pressure/Retention", z="Finishing/Threat", 
        color="Player", opacity=0.8,
        color_discrete_sequence=["#00ffcc", "#ff0066"],
        hover_data=["Player"]
    )
    
    # Make the plot look professional and dark-themed
    fig_3d.update_layout(
        scene=dict(
            bgcolor="#1e1e1e",
            xaxis=dict(title='Dim 1: Playmaking', backgroundcolor="#1e1e1e", gridcolor="gray"),
            yaxis=dict(title='Dim 2: Retention', backgroundcolor="#1e1e1e", gridcolor="gray"),
            zaxis=dict(title='Dim 3: Finishing', backgroundcolor="#1e1e1e", gridcolor="gray"),
        ),
        paper_bgcolor="#0e1117", 
        font=dict(color="white"),
        margin=dict(l=0, r=0, b=0, t=0) # Remove awkward white space
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    # --- SIMILARITY MATRIX ---
    st.markdown("---")
    st.subheader("🧬 Non-Linear DNA Similarity")
    
    # Calculate Similarity using the Neural Network's compressed Latent Vector
    mean_latent_A = np.mean(latent_A, axis=0)
    mean_latent_B = np.mean(latent_B, axis=0)
    deep_sim = 1 - cosine(mean_latent_A, mean_latent_B)
    
    # Keep the raw similarities as context
    spatial_sim = 1 - cosine(np.mean(sp_A, axis=0), np.mean(sp_B, axis=0))
    tactical_sim = 1 - cosine(np.mean(tac_A, axis=0), np.mean(tac_B, axis=0))
    
    sim_data = pd.DataFrame({
        "Similarity Dimension": ["Spatial (Raw Pitch Location)", "Semantic (Raw Match Actions)", "Deep Latent DNA (Autoencoder Vector)"],
        "Similarity Score (0-1)": [spatial_sim, tactical_sim, deep_sim]
    })
    
    st.table(sim_data.style.background_gradient(cmap='Purples', subset=['Similarity Score (0-1)']))