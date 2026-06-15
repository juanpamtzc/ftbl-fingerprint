# tests/test_models.py
import torch
import pytest
from src.ml_models import TacticalVAE

def test_vae_architecture():
    """Tests that the VAE accepts the correct input dimensions and outputs expected shapes."""
    input_dim = 400 # 384 spatial + 16 tactical
    latent_dim = 16
    batch_size = 32
    
    model = TacticalVAE(input_dim=input_dim, latent_dim=latent_dim)
    dummy_input = torch.randn(batch_size, input_dim)
    
    recon, mu, logvar = model(dummy_input)
    
    assert recon.shape == (batch_size, input_dim), "Reconstruction shape mismatch"
    assert mu.shape == (batch_size, latent_dim), "Latent space shape mismatch"
    assert logvar.shape == (batch_size, latent_dim), "Logvar shape mismatch"