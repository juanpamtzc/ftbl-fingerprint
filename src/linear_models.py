import pandas as pd
from sklearn.decomposition import PCA

def train_and_extract_pca(X_master, player_names, n_components=16):
    """Trains PCA and returns raw vectors and aggregated centroids."""
    print(f"Training {n_components}-Component PCA Baseline...")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_master)
    
    df_pca = pd.DataFrame(X_pca, columns=[f'PC_{i+1}' for i in range(n_components)])
    df_pca['Player'] = player_names
    
    # Build Centroids
    match_counts = df_pca['Player'].value_counts()
    robust_players = match_counts[match_counts >= 10].index
    df_robust = df_pca[df_pca['Player'].isin(robust_players)]
    df_centroids = df_robust.groupby('Player').mean()
    
    # FIX: Return the pca model as the first argument
    return pca, df_pca, df_centroids