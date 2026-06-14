import logging
from typing import List, Tuple
import numpy as np
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)


def run_mesh_convergence_study(match_coords: List[Tuple[np.ndarray, np.ndarray]]) -> None:
    """Verifies that spatial signatures reach stable asymptotic error limitations.

    Args:
        match_coords: Raw unbinned coordinate arrays from tracking events.
    """
    logger.info("Running spatial discretization mesh convergence validation...")
    mesh_sizes = [(3, 2), (6, 4), (12, 8), (24, 16), (30, 20), (60, 40)]
    
    print("\n--- GRID CONVERGENCE STUDY ---")
    print(f"{'Mesh (Nx x Ny)':<15} | {'Bins':<6} | {'Exp. Variance (3 PCs)':<22} | {'Train Error':<15} | {'Test Error':<15}")
    print("-" * 85)
    
    for nx, ny in mesh_sizes:
        match_matrices = []
        for x, y in match_coords:
            heatmap, _, _ = np.histogram2d(x, y, bins=[nx, ny], range=[[0, 120], [0, 80]])
            flat_vector = heatmap.flatten()
            flat_vector /= (flat_vector.sum() + 1e-9)
            match_matrices.append(flat_vector)
            
        X = np.array(match_matrices)
        midpoint = len(X) // 2
        X_train, X_test = X[:midpoint], X[midpoint:]
        
        pca = PCA(n_components=3)
        X_train_latent = pca.fit_transform(X_train)
        X_test_latent = pca.transform(X_test)
        
        train_err = mean_squared_error(X_train, pca.inverse_transform(X_train_latent))
        test_err = mean_squared_error(X_test, pca.inverse_transform(X_test_latent))
        exp_var = np.sum(pca.explained_variance_ratio_) * 100
        
        num_bins = nx * ny
        print(f"{str((nx, ny)):<15} | {num_bins:<6} | {exp_var:>15.2f}%       | {train_err:>13.6f} | {test_err:>13.6f}")


def validate_null_hypothesis(X_target: np.ndarray, X_control: np.ndarray) -> None:
    """Applies Wilcoxon testing to verify a signature captures the identity rather than the system.

    Args:
        X_target: Profile matrix belonging to target player (e.g., Messi).
        X_control: Profile matrix belonging to control player (e.g., Iniesta).
    """
    logger.info("Executing statistical cross-player verification...")
    
    pca = PCA(n_components=3)
    X_target_latent = pca.fit_transform(X_target)
    target_reconstructed = pca.inverse_transform(X_target_latent)
    
    X_control_latent = pca.transform(X_control)
    control_reconstructed = pca.inverse_transform(X_control_latent)
    
    target_errors = np.mean((X_target - target_reconstructed)**2, axis=1)
    control_errors = np.mean((X_control - control_reconstructed)**2, axis=1)
    
    print("\n--- STATISTICAL SIGNIFICANCE TEST ---")
    print(f"Target Mean Error:  {np.mean(target_errors):.6f} ± {np.std(target_errors):.6f}")
    print(f"Control Mean Error: {np.mean(control_errors):.6f} ± {np.std(control_errors):.6f}")
    
    stat, p_value = stats.wilcoxon(target_errors, control_errors)
    print(f"\nWilcoxon Signed-Rank Test p-value: {p_value:.2e}")
    
    if p_value < 0.05:
        print(">>> CONCLUSION: STATISTICALLY SIGNIFICANT (p < 0.05). Signature isolated successfully.\n")
    else:
        print(">>> CONCLUSION: NOT SIGNIFICANT. System fallback likely.\n")