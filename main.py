import pandas as pd
import numpy as np
import torch
import os
from sklearn.preprocessing import StandardScaler

from src.data_fetching import build_league_dataset
from src.ml_models import train_and_extract_vae
from src.linear_models import train_and_extract_pca

# ALL analytics and interpretability come from the V&V module
from src.verification_validation import (
    evaluate_manifold_stability,
    run_vae_diagnostics,
    interpret_vae_manifold,
    interpret_pca_loadings
)

def main():
    data_path = "tactical_database.parquet"
    if not os.path.exists(data_path):
        print("Data not found. Initiating Fetch & Kinematic Extraction...")
        df = build_league_dataset(output_path=data_path)
    else:
        print("Loading existing dataset...")
        df = pd.read_parquet(data_path)

    tactical_keys = [
        'passes_attempted', 'pass_completion_pct', 'shots', 'goals', 'assists', 
        'carries', 'dribbles', 'fouls_won', 'interceptions', 'tackles', 'clearances',
        'avg_action_velocity', 'max_action_velocity', 'explosive_bursts', 
        'total_action_distance', 'avg_deceleration'
    ]

    # Prepare Data
    X_spatial = np.stack(df['spatial_vector'].values)
    X_tactical = np.stack(df['tactical_vector'].values)
    
    scaler = StandardScaler()
    X_tactical_scaled = scaler.fit_transform(X_tactical)
    X_master = np.hstack((X_spatial, X_tactical_scaled))
    player_names = df['player_name'].values
    
    print("\n--- 1. EVALUATING PCA BASELINE ---")
    pca_model, df_pca, df_pca_centroids = train_and_extract_pca(X_master, player_names, n_components=16)
    _, _, pca_ratio = evaluate_manifold_stability(df_pca)
    print(f"PCA Separation Ratio: {pca_ratio:.2f}x")
    
    # Validation/Interpretability
    interpret_pca_loadings(pca_model, tactical_keys)
    
    print("\n--- 2. VAE DIAGNOSTICS & TRAINING ---")
    # Validation/Diagnostics
    run_vae_diagnostics(X_master, player_names, input_dim=X_master.shape[1])
    
    X_tensor = torch.FloatTensor(X_master)
    vae_model, df_vae, df_vae_centroids = train_and_extract_vae(X_tensor, player_names, input_dim=X_master.shape[1])
    _, _, vae_ratio = evaluate_manifold_stability(df_vae)
    print(f"VAE Separation Ratio: {vae_ratio:.2f}x")
    
    # Validation/Interpretability
    interpret_vae_manifold(vae_model, scaler, tactical_keys, latent_dim=16)
    
    print("\n--- 3. EXPORTING CENTROIDS FOR APP ---")
    df_pca_centroids.to_csv("pca_centroids.csv")
    df_vae_centroids.to_csv("vae_centroids.csv")
    print("Export Complete! Run `streamlit run app.py` to view the engine.")

if __name__ == "__main__":
    main()