import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

def evaluate_manifold_stability(df_latent, min_matches=10):
    """Executes the Split-Half Centroid test to calculate the Separation Ratio."""
    match_counts = df_latent['Player'].value_counts()
    robust_players = match_counts[match_counts >= min_matches].index.tolist()
    df_robust = df_latent[df_latent['Player'].isin(robust_players)]
    
    centroids_A = {}
    centroids_B = {}
    
    for player in robust_players:
        player_data = df_robust[df_robust['Player'] == player].drop(columns=['Player']).values
        np.random.shuffle(player_data)
        
        midpoint = len(player_data) // 2
        centroids_A[player] = np.mean(player_data[:midpoint], axis=0)
        centroids_B[player] = np.mean(player_data[midpoint:], axis=0)
        
    same_dist = []
    diff_dist = []
    
    for player in robust_players:
        same_dist.append(euclidean(centroids_A[player], centroids_B[player]))
        
    for i, p1 in enumerate(robust_players):
        for j, p2 in enumerate(robust_players):
            if i != j:
                diff_dist.append(euclidean(centroids_A[p1], centroids_B[p2]))
                
    mean_same = np.mean(same_dist)
    mean_diff = np.mean(diff_dist)
    ratio = mean_diff / mean_same
    
    return mean_same, mean_diff, ratio