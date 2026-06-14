import logging
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsbombpy import sb

logger = logging.getLogger(__name__)

def get_raw_match_coordinates(
    competition_id: int = 11, season_id: int = 22, player_name: str = "Lionel Andrés Messi Cuccittini"
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Fetches matches and returns a list of raw (X,Y) coordinate arrays per match.

    Args:
        competition_id: The StatsBomb competition ID (Default: 11 for La Liga).
        season_id: The StatsBomb season ID (Default: 22 for 2011/12).
        player_name: The target player's full name.

    Returns:
        A list of tuples, where each tuple contains (X_coordinates, Y_coordinates) for a match.
    """
    logger.info(f"Fetching raw spatial coordinates for {player_name}...")
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    barca_matches = matches[(matches["home_team"] == "Barcelona") | (matches["away_team"] == "Barcelona")]
    match_ids = barca_matches["match_id"].tolist()

    match_coords = []

    for m_id in match_ids:
        try:
            events = sb.events(match_id=m_id)
            if "player" not in events.columns or "location" not in events.columns:
                continue

            player_events = events[(events["player"] == player_name) & (events["location"].notnull())]
            if len(player_events) < 15:
                continue

            locations = np.array(player_events["location"].tolist())
            match_coords.append((locations[:, 0], locations[:, 1]))
        except Exception as e:
            logger.debug(f"Failed to process match {m_id}: {e}")
            continue

    logger.info(f"Successfully extracted coordinates for {len(match_coords)} matches.")
    return match_coords


def get_coupled_signatures(
    competition_id: int = 11,
    season_id: int = 22,
    mesh: Tuple[int, int] = (24, 16),
    player_name: str = "Lionel Andrés Messi Cuccittini"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Extracts both spatial and semantic tactical features, combining them into a coupled matrix.

    Args:
        competition_id: The StatsBomb competition ID.
        season_id: The StatsBomb season ID.
        mesh: A tuple representing the (X, Y) bins for the spatial grid.
        player_name: The target player's full name.

    Returns:
        A tuple containing (Spatial Matrix, Scaled Tactical Matrix, Coupled Matrix, Feature Names).
    """
    logger.info(f"Extracting Dual-Signatures for {player_name}...")
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    barca_matches = matches[(matches["home_team"] == "Barcelona") | (matches["away_team"] == "Barcelona")]
    match_ids = barca_matches["match_id"].tolist()

    spatial_matrices, tactical_matrices = [], []
    tactical_keys = [
        "passes_attempted", "pass_completion_pct", "shots",
        "goals", "assists", "carries", "dribbles", "fouls_won"
    ]

    for m_id in match_ids:
        try:
            events = sb.events(match_id=m_id)
            if "player" not in events.columns:
                continue

            player_events = events[events["player"] == player_name]
            if len(player_events) < 15:
                continue

            # 1. SPATIAL SIGNATURE
            loc_events = player_events[player_events["location"].notnull()]
            locations = np.array(loc_events["location"].tolist())
            heatmap, _, _ = np.histogram2d(
                locations[:, 0], locations[:, 1], bins=[mesh[0], mesh[1]], range=[[0, 120], [0, 80]]
            )
            spatial_vector = heatmap.flatten()
            spatial_vector /= (spatial_vector.sum() + 1e-9)

            # 2. TACTICAL ACTION SIGNATURE
            tactical_stats = {key: 0.0 for key in tactical_keys}
            tactical_stats["passes_attempted"] = len(player_events[player_events["type"] == "Pass"])

            if "pass_outcome" in player_events.columns and tactical_stats["passes_attempted"] > 0:
                completed = len(player_events[(player_events["type"] == "Pass") & (player_events["pass_outcome"].isnull())])
                tactical_stats["pass_completion_pct"] = completed / tactical_stats["passes_attempted"]

            tactical_stats["shots"] = len(player_events[player_events["type"] == "Shot"])
            if "shot_outcome" in player_events.columns:
                tactical_stats["goals"] = len(player_events[(player_events["type"] == "Shot") & (player_events["shot_outcome"] == "Goal")])

            if "pass_goal_assist" in player_events.columns:
                tactical_stats["assists"] = len(player_events[(player_events["type"] == "Pass") & (player_events["pass_goal_assist"] == True)])

            tactical_stats["carries"] = len(player_events[player_events["type"] == "Carry"])
            tactical_stats["dribbles"] = len(player_events[player_events["type"] == "Dribble"])
            tactical_stats["fouls_won"] = len(player_events[player_events["type"] == "Foul Won"])

            tactical_vector = np.array([tactical_stats[k] for k in tactical_keys])

            spatial_matrices.append(spatial_vector)
            tactical_matrices.append(tactical_vector)

        except Exception as e:
            logger.debug(f"Failed to process match {m_id}: {e}")
            continue

    X_spatial = np.array(spatial_matrices)
    X_tactical_raw = np.array(tactical_matrices)

    # Standardization
    scaler = StandardScaler()
    X_tactical_scaled = scaler.fit_transform(X_tactical_raw)
    X_coupled = np.hstack((X_spatial, X_tactical_scaled))

    logger.info(f"Extraction Complete. Coupled Shape: {X_coupled.shape}")
    return X_spatial, X_tactical_scaled, X_coupled, tactical_keys