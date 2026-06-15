# tests/test_linear_models.py
import numpy as np
import pandas as pd
from src.linear_models import train_and_extract_pca

def test_pca_extraction():
    """Tests that PCA correctly reduces dimensions and groups player centroids."""
    n_samples = 50
    input_dim = 400 # 384 spatial + 16 tactical
    n_components = 16
    
    # Generate dummy data
    X_dummy = np.random.rand(n_samples, input_dim)
    
    # Create 5 fake players, each with 10 matches (50 samples total)
    player_names = np.array([f"Player_{i}" for i in range(5) for _ in range(10)])
    
    pca_model, df_pca, df_centroids = train_and_extract_pca(
        X_master=X_dummy, 
        player_names=player_names, 
        n_components=n_components
    )
    
    # 1. Check if PCA reduced the matrix correctly
    assert df_pca.shape[1] == n_components + 1 # 16 components + 'Player' column
    
    # 2. Check if centroids grouped properly (should be 5 unique players)
    assert len(df_centroids) == 5
    
    # 3. Check centroid dimensions match PCA components
    assert df_centroids.shape[1] == n_components