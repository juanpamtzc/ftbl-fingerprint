import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
from statsbombpy import sb
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from src.linear_models import train_and_extract_pca

# Import model architecture for validation testing
from src.ml_models import TacticalVAE, vae_loss_fn

# ----------------------------------------------------------------
# 1. MANIFOLD STABILITY (SPLIT-HALF CENTROIDS)
# ----------------------------------------------------------------
def evaluate_manifold_stability(df_latent, min_matches=10):
    """Executes the Split-Half Centroid test to calculate the Separation Ratio."""
    match_counts = df_latent['Player'].value_counts()
    robust_players = match_counts[match_counts >= min_matches].index.tolist()
    df_robust = df_latent[df_latent['Player'].isin(robust_players)]
    
    centroids_A, centroids_B = {}, {}
    for player in robust_players:
        player_data = df_robust[df_robust['Player'] == player].drop(columns=['Player']).values
        np.random.shuffle(player_data)
        
        midpoint = len(player_data) // 2
        centroids_A[player] = np.mean(player_data[:midpoint], axis=0)
        centroids_B[player] = np.mean(player_data[midpoint:], axis=0)
        
    same_dist = [euclidean(centroids_A[p], centroids_B[p]) for p in robust_players]
    diff_dist = []
    
    for i, p1 in enumerate(robust_players):
        for j, p2 in enumerate(robust_players):
            if i != j:
                diff_dist.append(euclidean(centroids_A[p1], centroids_B[p2]))
                
    mean_same = np.mean(same_dist)
    mean_diff = np.mean(diff_dist)
    ratio = mean_diff / mean_same
    
    return mean_same, mean_diff, ratio

# ----------------------------------------------------------------
# 2. VAE DIAGNOSTICS & ELBOW METHOD
# ----------------------------------------------------------------
def run_vae_diagnostics(X_master, player_names, input_dim):
    """Runs the Elbow Method and Convergence Analysis, saving plots to disk."""
    print("\n--- RUNNING VAE DIAGNOSTICS (Elbow & Convergence) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    unique_players = np.unique(player_names)
    train_players, unseen_players = train_test_split(unique_players, test_size=0.15, random_state=42)
    
    train_mask = np.isin(player_names, train_players)
    unseen_mask = np.isin(player_names, unseen_players)
    
    X_train_raw, X_seen_test_raw = train_test_split(X_master[train_mask], test_size=0.20, random_state=42)
    X_unseen_test_raw = X_master[unseen_mask]
    
    X_train_tensor = torch.tensor(X_train_raw, dtype=torch.float32).to(device)
    X_seen_tensor = torch.tensor(X_seen_test_raw, dtype=torch.float32).to(device)
    X_unseen_tensor = torch.tensor(X_unseen_test_raw, dtype=torch.float32).to(device)
    
    train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=256, shuffle=True)
    
    latent_grid = [4, 8, 12, 16, 24, 32]
    seen_losses, unseen_losses = [], []
    
    print("Executing Elbow Method...")
    for dim in latent_grid:
        model = TacticalVAE(input_dim, latent_dim=dim).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(50):
            for batch in train_loader:
                optimizer.zero_grad()
                recon, mu, logvar = model(batch[0])
                loss = vae_loss_fn(recon, batch[0], mu, logvar)
                loss.backward()
                optimizer.step()
                
        model.eval()
        with torch.no_grad():
            r_seen, _, _ = model(X_seen_tensor)
            seen_losses.append(nn.functional.mse_loss(r_seen, X_seen_tensor).item())
            r_unseen, _, _ = model(X_unseen_tensor)
            unseen_losses.append(nn.functional.mse_loss(r_unseen, X_unseen_tensor).item())
            
    plt.figure(figsize=(10, 6))
    plt.plot(latent_grid, seen_losses, marker='o', label='Seen Players')
    plt.plot(latent_grid, unseen_losses, marker='s', linestyle='--', label='Unseen Players')
    plt.title('VAE Elbow Method: Reconstruction Loss vs Latent Dimensions')
    plt.xlabel('Latent Dimensions')
    plt.ylabel('MSE Reconstruction Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('vae_elbow_curve.png')
    plt.close()
    print("Saved 'vae_elbow_curve.png'")

# ----------------------------------------------------------------
# 3. INTERPRETABILITY TOOLS (LATENT TRAVERSAL & PCA LOADINGS)
# ----------------------------------------------------------------
def interpret_vae_manifold(model, scaler, tactical_keys, latent_dim=16):
    """Executes Latent Traversal to see what each dimension learned."""
    print("\n--- VAE LATENT TRAVERSAL REPORT ---")
    device = next(model.parameters()).device
    model.eval()
    
    report = []
    with torch.no_grad():
        for dim in range(latent_dim):
            base_z = torch.zeros(1, latent_dim).to(device)
            
            z_low, z_high = base_z.clone(), base_z.clone()
            z_low[0, dim] = -3.0
            z_high[0, dim] = 3.0
            
            recon_low = model.decoder(z_low).cpu().numpy()[0][96:]
            recon_high = model.decoder(z_high).cpu().numpy()[0][96:]
            
            tac_low_real = scaler.inverse_transform([recon_low])[0]
            tac_high_real = scaler.inverse_transform([recon_high])[0]
            
            deltas = tac_high_real - tac_low_real
            top_idx = np.argmax(np.abs(deltas))
            
            stat_name = tactical_keys[top_idx]
            direction = "INCREASES" if deltas[top_idx] > 0 else "DECREASES"
            report.append(f"Dim {dim+1}: Controls '{stat_name}' ({direction} from {tac_low_real[top_idx]:.1f} to {tac_high_real[top_idx]:.1f})")
            
    with open("vae_interpretability_report.txt", "w") as f:
        f.write("\n".join(report))
    print("Saved 'vae_interpretability_report.txt'")


def interpret_pca_loadings(pca_model, tactical_keys):
    """Analyzes the PCA components to see which features drive the variance."""
    print("\n--- PCA LOADINGS REPORT ---")
    tactical_components = pca_model.components_[:, 96:] 
    
    report = []
    for i in range(min(5, tactical_components.shape[0])):
        component = tactical_components[i]
        top_indices = np.argsort(np.abs(component))[::-1][:3]
        
        report.append(f"Principal Component {i+1} Driver Features:")
        for idx in top_indices:
            weight = component[idx]
            impact = "Positive" if weight > 0 else "Negative"
            report.append(f"  - {tactical_keys[idx]}: {weight:.3f} ({impact} Correlation)")
        report.append("")
        
    with open("pca_interpretability_report.txt", "w") as f:
        f.write("\n".join(report))
    print("Saved 'pca_interpretability_report.txt'")


# ----------------------------------------------------------------
# 4. MESH CONVERGENCE ANALYSIS
# ----------------------------------------------------------------
def run_mesh_convergence(match_limit=30):
    """
    Tests different grid resolutions to find the optimal balance between 
    spatial nuance and vector sparsity.
    """
    print(f"Starting Mesh Convergence Analysis on {match_limit} matches...")
    
    # 1. Fetch a subset of events
    matches = sb.matches(competition_id=11, season_id=27).head(match_limit)
    events_list = []
    for m_id in tqdm(matches['match_id'], desc="Fetching Match Events"):
        try:
            ev = sb.events(match_id=m_id)
            valid = ev.dropna(subset=['location', 'player']).copy()
            if len(valid) > 0:
                events_list.append(valid)
        except:
            continue
            
    if not events_list:
        print("Failed to pull events for convergence test.")
        return
        
    df_events = pd.concat(events_list, ignore_index=True)
    df_events['x'] = df_events['location'].apply(lambda loc: loc[0])
    df_events['y'] = df_events['location'].apply(lambda loc: loc[1])
    
    # 2. Define the meshes to test (Aspect ratio 1.5)
    meshes = [[12, 8], [15, 10], [24, 16], [30, 20], [45, 30]]
    ratios = []
    
    for bins in meshes:
        n_bins = bins[0] * bins[1]
        print(f"\nEvaluating Mesh: {bins[0]}x{bins[1]} ({n_bins} Spatial Dimensions)")
        
        dataset = []
        for player, group in df_events.groupby('player'):
            if len(group) < 15:
                continue
                
            heatmap, _, _ = np.histogram2d(
                group['x'], group['y'], 
                bins=bins, range=[[0, 120], [0, 80]]
            )
            spatial_vector = heatmap.flatten()
            spatial_vector /= (spatial_vector.sum() + 1e-9)
            
            # Dummy tactical vector for pure spatial separation test
            tactical_vector = [len(group)] * 16 
            
            dataset.append({
                'Player': player,
                'Spatial': spatial_vector.tolist(),
                'Tactical': tactical_vector
            })
            
        df_mesh = pd.DataFrame(dataset)
        X_spatial = np.stack(df_mesh['Spatial'].values)
        X_tactical = StandardScaler().fit_transform(np.stack(df_mesh['Tactical'].values))
        X_master = np.hstack((X_spatial, X_tactical))
        
        # Run Baseline Separation Test (PCA)
        pca_model, df_pca, df_pca_centroids = train_and_extract_pca(X_master, df_mesh['Player'].values, n_components=16)
        
        try:
            _, _, ratio = evaluate_manifold_stability(df_pca, min_matches=2) 
            ratios.append(ratio)
            print(f"Separation Ratio: {ratio:.2f}x")
        except Exception as e:
            print(f"Error calculating stability: {e}")
            ratios.append(0)
            
    # 3. Plot the Convergence Curve
    plt.figure(figsize=(10, 6))
    x_labels = [f"{b[0]}x{b[1]}\n({b[0]*b[1]} bins)" for b in meshes]
    plt.plot(x_labels, ratios, marker='o', linewidth=2, color='red')
    plt.title('Mesh Convergence Analysis: Spatial Resolution vs Latent Stability')
    plt.xlabel('Grid Resolution (Bins)')
    plt.ylabel('Separation Ratio (Higher is better)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('mesh_convergence.png')
    print("\nSaved 'mesh_convergence.png'.")