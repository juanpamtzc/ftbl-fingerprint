# tests/test_validation.py
import pandas as pd
import numpy as np
from src.verification_validation import evaluate_manifold_stability

def test_manifold_stability_logic():
    """Tests the distance calculation between intra-player and inter-player matrices."""
    
    # Create a perfectly stable dummy dataset (Players are identical to themselves, different from others)
    data = {
        'Player': ['Messi', 'Messi', 'Ronaldo', 'Ronaldo'],
        'Dim_1': [1.0, 1.01, 10.0, 10.01],
        'Dim_2': [1.0, 1.05, 10.0, 10.05]
    }
    df_latent = pd.DataFrame(data)
    
    same_dist, diff_dist, ratio = evaluate_manifold_stability(df_latent, min_matches=2)
    
    # 1. Assert distances were calculated
    assert same_dist > 0
    assert diff_dist > 0
    
    # 2. Assert separation ratio is mathematically valid (Diff should be much larger than Same here)
    assert ratio > 1.0 
    assert isinstance(ratio, float)