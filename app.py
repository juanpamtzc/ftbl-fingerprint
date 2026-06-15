import streamlit as st
import pandas as pd
from scipy.spatial.distance import cdist

st.set_page_config(page_title="Scouting Engine", layout="wide")

@st.cache_data
def load_models():
    pca_df = pd.read_csv("pca_centroids.csv", index_col="Player")
    vae_df = pd.read_csv("vae_centroids.csv", index_col="Player")
    return pca_df, vae_df

pca_centroids, vae_centroids = load_models()
players = sorted(pca_centroids.index.tolist())

st.title("⚽ Intelligent Latent Scouting Engine")
st.markdown("""
**Model Selection Context:** This engine calculates high-dimensional split-half centroids to bypass single-match variance. You can toggle between a **16-Component PCA** and a **16D Variational Autoencoder (VAE)**. Both models achieve a phenomenal >2.30x mathematical separation ratio.
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Search Parameters")
    target_player = st.selectbox("Target Player:", players)
    model_choice = st.radio("Underlying Architecture:", ["PCA (Linear Baseline)", "VAE (Non-Linear Manifold)"])
    n_results = st.slider("Replacements to retrieve:", 1, 10, 5)

with col2:
    st.subheader(f"Top {n_results} Tactical Twins")
    
    # Select the requested manifold
    active_df = pca_centroids if model_choice == "PCA (Linear Baseline)" else vae_centroids
    
    # Distance Calculation
    target_vector = active_df.loc[target_player].values.reshape(1, -1)
    distances = cdist(target_vector, active_df.values, metric='euclidean').flatten()
    
    results = pd.DataFrame({
        'Player': active_df.index,
        'Latent Distance': distances
    }).sort_values('Latent Distance')
    
    # Drop self and show
    results = results[results['Player'] != target_player].head(n_results)
    st.dataframe(results.style.background_gradient(cmap='Blues_r', subset=['Latent Distance']), 
                 use_container_width=True, hide_index=True)