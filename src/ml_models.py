import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from typing import Tuple, List

class TacticalVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(TacticalVAE, self).__init__()
        self.encoder_fc = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.SiLU(),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.SiLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.LayerNorm(32), nn.SiLU(),
            nn.Linear(32, 64), nn.LayerNorm(64), nn.SiLU(),
            nn.Linear(64, input_dim)
        )
        
    def encode(self, x):
        h = self.encoder_fc(x)
        return self.fc_mu(h), self.fc_logvar(h)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def vae_loss_fn(recon_x, x, mu, logvar, beta=0.005):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return recon_loss + beta * kl_loss

def train_and_extract_vae(X_tensor, player_names, input_dim, latent_dim=16, epochs=500):
    """Trains the VAE and returns both the raw latent vectors and the aggregated centroids."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TacticalVAE(input_dim, latent_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    X_tensor = X_tensor.to(device)
    
    print(f"Training {latent_dim}D VAE on {device} for {epochs} epochs...")
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(X_tensor)
        loss = vae_loss_fn(recon_batch, X_tensor, mu, logvar, beta=0.005)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        latent_mu, _ = model.encode(X_tensor)
        
    # Build Raw Latent DataFrame
    df_latent = pd.DataFrame(latent_mu.cpu().numpy(), columns=[f'Dim_{i+1}' for i in range(latent_dim)])
    df_latent['Player'] = player_names
    
    # Build Centroids DataFrame
    match_counts = df_latent['Player'].value_counts()
    robust_players = match_counts[match_counts >= 10].index
    df_robust = df_latent[df_latent['Player'].isin(robust_players)]
    df_centroids = df_robust.groupby('Player').mean()
    
    return df_latent, df_centroids

def explain_autoencoder_black_box(
    ae_model: nn.Module, X_coupled: np.ndarray, tactical_features: List[str]
) -> None:
    """Performs latent space traversal to decode the deep network dimensions.

    Args:
        ae_model: The trained PyTorch Autoencoder module.
        X_coupled: Standardized matrix used for matching indices.
        tactical_features: Ordered labels matching semantic columns.
    """
    logger.info("Executing non-linear latent traversal analysis...")
    print("\n--- DECODING THE AUTOENCODER (LATENT TRAVERSAL) ---\n")
    
    ae_model.eval()
    X_tensor = torch.FloatTensor(X_coupled)
    
    with torch.no_grad():
        _, latent_space = ae_model(X_tensor)
        mean_latent = torch.mean(latent_space, dim=0)
        baseline_output = ae_model.decoder(mean_latent).detach().numpy()
        
        num_spatial = X_coupled.shape[1] - len(tactical_features)
        baseline_tactical = baseline_output[num_spatial:]
        
        perturbation_amount = 2.0 
        
        for dim in range(3):
            print(f"### What does Latent Dimension {dim+1} control?")
            perturbed_latent = mean_latent.clone()
            perturbed_latent[dim] += perturbation_amount
            
            perturbed_output = ae_model.decoder(perturbed_latent).detach().numpy()
            perturbed_tactical = perturbed_output[num_spatial:]
            
            deltas = perturbed_tactical - baseline_tactical
            impacts = list(zip(tactical_features, deltas))
            impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            
            for feature, delta in impacts[:4]: 
                direction = "INCREASES" if delta > 0 else "DECREASES"
                print(f"  -> Pushing Dim {dim+1} {direction} {feature:<20} by {abs(delta):>5.2f} std devs")
            print()