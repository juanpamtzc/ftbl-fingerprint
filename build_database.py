"""
Tactical Engine - Offline Database Builder
Fetches all target players, trains the Master Autoencoder, and saves the latent DNA to a CSV.
"""

import logging
import pandas as pd
import numpy as np
import torch
from src.data_fetching import get_coupled_signatures
from src.ml_models import train_autoencoder

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# 1. Define our target database scope
SCOUTING_TARGETS = [
    {"player": "Lionel Andrés Messi Cuccittini", "comp": 11, "season": 22, "season_name": "2011/12"},
    {"player": "Andrés Iniesta Luján", "comp": 11, "season": 22, "season_name": "2011/12"},
    {"player": "Xavier Hernández Creus", "comp": 11, "season": 22, "season_name": "2011/12"}, # Xavi
    {"player": "Lionel Andrés Messi Cuccittini", "comp": 11, "season": 27, "season_name": "2015/16"},
    {"player": "Neymar da Silva Santos Junior", "comp": 11, "season": 27, "season_name": "2015/16"},
    {"player": "Luis Alberto Suárez Díaz", "comp": 11, "season": 27, "season_name": "2015/16"}
]

def build_database():
    logger.info("Starting Offline Database Generation...")
    
    all_coupled = []
    all_metadata = []
    
    tactical_feature_names = None
    
    # 2. Fetch all raw data from StatsBomb
    for target in SCOUTING_TARGETS:
        logger.info(f"Fetching {target['player']} for {target['season_name']}...")
        try:
            sp, raw_tac, tac_scaled, coup, features = get_coupled_signatures(
                competition_id=target['comp'], 
                season_id=target['season'], 
                player_name=target['player']
            )
            
            if len(coup) == 0:
                continue
                
            tactical_feature_names = features
            
            # Save the matrices and metadata to assemble later
            all_coupled.append(coup)
            
            # Create a metadata row for every single match this player played
            for i in range(len(coup)):
                meta = {
                    "Player": target['player'],
                    "Season": target['season_name'],
                }
                # Add the raw tactical stats (Goals, Passes, etc.)
                for j, feat in enumerate(features):
                    meta[feat] = raw_tac[i, j]
                    
                # Add the spatial bins (384 columns for the pitch heatmap)
                for j in range(sp.shape[1]):
                    meta[f"spatial_{j}"] = sp[i, j]
                    
                all_metadata.append(meta)
                
        except Exception as e:
            logger.error(f"Failed to fetch {target['player']}: {e}")
            
    # 3. Stack everything into one giant Universe Matrix
    X_master = np.vstack(all_coupled)
    logger.info(f"\nConstructed Master Matrix with {X_master.shape[0]} total matches.")
    
    # 4. Train the ONE Shared Latent Space Autoencoder
    logger.info("Training the Universal Autoencoder...")
    shared_ae = train_autoencoder(X_master, latent_dim=3, epochs=1000)
    
    # 5. Extract the 3D Coordinates for every match
    logger.info("Extracting Latent DNA...")
    shared_ae.eval()
    with torch.no_grad():
        _, latent_master = shared_ae(torch.FloatTensor(X_master))
        latent_master = latent_master.numpy()
        
    # 6. Assemble the final Database
    df_database = pd.DataFrame(all_metadata)
    df_database["Dim_1_Playmaking"] = latent_master[:, 0]
    df_database["Dim_2_Retention"] = latent_master[:, 1]
    df_database["Dim_3_Finishing"] = latent_master[:, 2]
    
    # Reorder columns so dimensions are at the front
    cols = ["Player", "Season", "Dim_1_Playmaking", "Dim_2_Retention", "Dim_3_Finishing"] + tactical_feature_names + [f"spatial_{i}" for i in range(384)]
    df_database = df_database[cols]
    
    # 7. Save to CSV
    filename = "tactical_database.csv"
    df_database.to_csv(filename, index=False)
    logger.info(f"\nSUCCESS! Database saved to {filename}")

if __name__ == "__main__":
    build_database()