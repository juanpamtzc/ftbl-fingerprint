import logging
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

logger = logging.getLogger(__name__)


class TacticalAutoencoder(nn.Module):
    """Deep Convolutional/Linear Autoencoder to compress player action spaces."""
    
    def __init__(self, input_dim: int, latent_dim: int = 3):
        """Initializes the autoencoder.
        
        Args:
            input_dim: The number of features in the coupled matrix.
            latent_dim: The desired bottleneck dimensionality.
        """
        super(TacticalAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 128), nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


def train_autoencoder(X_coupled: np.ndarray, latent_dim: int = 3, epochs: int = 1000) -> TacticalAutoencoder:
    """Trains the PyTorch Autoencoder on the coupled feature matrix.

    Args:
        X_coupled: The standardized NumPy array combining spatial and tactical data.
        latent_dim: The size of the bottleneck layer.
        epochs: Number of training iterations.

    Returns:
        The trained TacticalAutoencoder model.
    """
    logger.info(f"Initializing PyTorch Autoencoder training for {epochs} epochs...")
    X_tensor = torch.FloatTensor(X_coupled)
    input_dim = X_coupled.shape[1]
    
    model = TacticalAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        reconstructed, _ = model(X_tensor)
        loss = criterion(reconstructed, X_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 250 == 0:
            logger.debug(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
            
    logger.info("Training complete.")
    return model

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