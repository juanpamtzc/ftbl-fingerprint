import pandas as pd
import numpy as np
import torch
import os
from sklearn.preprocessing import StandardScaler

from src.data_fetching import build_league_dataset
from src.ml_models import train_and_extract_vae
from src.linear_models import train_and_extract_pca
from src.verification_validation import evaluate_manifold_stability

def main():
    data_path = "tactical_database.parquet"
    if not os.path.exists(data_path):
        print("Data not found. Initiating Fetch & Kinematic Extraction...")
        df = build_league_dataset(output_path=data_path)
    else:
        print("Loading existing dataset...")
        df = pd.read_parquet(data_path)

    # Prepare Data
    X_spatial = np.stack(df['spatial_vector'].values)
    X_tactical = np.stack(df['tactical_vector'].values)
    X_tactical_scaled = StandardScaler().fit_transform(X_tactical)
    
    X_master = np.hstack((X_spatial, X_tactical_scaled))
    player_names = df['player_name'].values
    
    print("\n--- 1. EVALUATING PCA BASELINE ---")
    df_pca, df_pca_centroids = train_and_extract_pca(X_master, player_names, n_components=16)
    _, _, pca_ratio = evaluate_manifold_stability(df_pca)
    print(f"PCA Separation Ratio: {pca_ratio:.2f}x")
    
    print("\n--- 2. EVALUATING VAE MANIFOLD ---")
    X_tensor = torch.FloatTensor(X_master)
    df_vae, df_vae_centroids = train_and_extract_vae(X_tensor, player_names, input_dim=X_master.shape[1])
    _, _, vae_ratio = evaluate_manifold_stability(df_vae)
    print(f"VAE Separation Ratio: {vae_ratio:.2f}x")
    
    print("\n--- 3. EXPORTING CENTROIDS FOR APP ---")
    df_pca_centroids.to_csv("pca_centroids.csv")
    df_vae_centroids.to_csv("vae_centroids.csv")
    print("Export Complete! Run `streamlit run app.py` to view the engine.")

if __name__ == "__main__":
    main()