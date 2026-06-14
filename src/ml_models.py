import logging
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