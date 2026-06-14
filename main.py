"""
Tactical Engine - Orchestration Pipeline
Executes linear tracking baselines, non-linear deep learning, and grid validation.
"""

import logging
from sklearn.metrics import mean_squared_error
from src.data_fetching import get_raw_match_coordinates, get_coupled_signatures
from src.linear_models import run_pca_analysis
from src.ml_models import train_autoencoder, explain_autoencoder_black_box
from src.verification_validation import run_mesh_convergence_study, validate_null_hypothesis

# Global logging configurations
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("STARTING PRODUCTION TACTICAL PROFILE ENGINE...")
    
    # 1. Structural Verification Block
    raw_coords = get_raw_match_coordinates()
    run_mesh_convergence_study(raw_coords)
    
    # 2. Vector Extraction Block
    X_spatial_m, X_tactical_m, X_coupled_m, features = get_coupled_signatures(player_name="Lionel Andrés Messi Cuccittini")
    X_spatial_i, _, _, _ = get_coupled_signatures(player_name="Andrés Iniesta Luján")
    
    # 3. Target Uniqueness Validation Block
    validate_null_hypothesis(X_target=X_spatial_m, X_control=X_spatial_i)
    
    # 4. Linear Engine Execution (PCA Baseline Analysis)
    _, _, pca_mse = run_pca_analysis(
        X_coupled=X_coupled_m, 
        spatial_dim=X_spatial_m.shape[1], 
        tactical_features=features
    )
    
    # 5. Non-Linear Deep Learning Network Block
    trained_ae = train_autoencoder(X_coupled_m, latent_dim=3, epochs=1000)
    
    # Direct Error Comparison Assessment
    import torch
    X_tensor = torch.FloatTensor(X_coupled_m)
    ae_reconstructed, _ = trained_ae(X_tensor)
    ae_mse = float(mean_squared_error(X_coupled_m, ae_reconstructed.detach().numpy()))
    
    print("--- COMPRESSION MATCHUP RESULT ---")
    print(f"Linear (PCA) Error:       {pca_mse:.6f}")
    print(f"Non-Linear (AE) Error:    {ae_mse:.6f}")
    improvement = ((pca_mse - ae_mse) / pca_mse) * 100
    print(f">> Deep Autoencoder structural information loss reduction: {improvement:.2f}%\n")
    
    # 6. Interpretability Breakdown Block
    explain_autoencoder_black_box(trained_ae, X_coupled_m, features)
    
    logger.info("ENGINE EXECUTION PIPELINE FULLY COMPLETED.")


if __name__ == "__main__":
    main()