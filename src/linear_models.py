"""Linear baseline models for tactical signature analysis."""

import logging
from typing import List, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)


def run_pca_analysis(
    X_coupled: np.ndarray, spatial_dim: int, tactical_features: List[str], n_components: int = 3
) -> Tuple[PCA, np.ndarray, float]:
    """Fits PCA on the coupled matrix, prints feature loadings, and returns the baseline.

    Args:
        X_coupled: Combined spatial and tactical array.
        spatial_dim: The number of spatial columns (bins) in the array.
        tactical_features: List of tactical action strings.
        n_components: Number of principal components to extract.

    Returns:
        A tuple of (fitted PCA object, latent representation array, reconstruction MSE error).
    """
    logger.info(f"Fitting PCA baseline with {n_components} components...")
    
    pca = PCA(n_components=n_components)
    X_latent = pca.fit_transform(X_coupled)
    X_reconstructed = pca.inverse_transform(X_latent)
    
    pca_mse = float(mean_squared_error(X_coupled, X_reconstructed))
    
    print("\n--- UNIFIED LINEAR SIGNATURE ANALYSIS (PCA) ---")
    print(f"Explained Variance (Top 3 Coupled Components): {np.sum(pca.explained_variance_ratio_)*100:.2f}%\n")
    
    # Extract feature loadings only for tactical actions (skipping spatial columns)
    tactical_loadings = pca.components_[:, spatial_dim:]
    
    for i in range(n_components):
        print(f"### Component {i+1} Tactical Drivers (Top Weights):")
        feature_weights = list(zip(tactical_features, tactical_loadings[i]))
        feature_weights.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for feature, weight in feature_weights[:4]:
            direction = "INCREASES" if weight > 0 else "DECREASES"
            print(f"  -> {feature:<20}: {weight:>6.3f} ({direction} this style)")
        print()
        
    return pca, X_latent, pca_mse