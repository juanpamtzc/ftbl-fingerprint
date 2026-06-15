import pandas as pd
import numpy as np
from statsbombpy import sb
from tqdm import tqdm
import os

def build_league_dataset(comp_id=11, season_id=27, output_path="tactical_database.parquet"):
    """Fetches every player-match vector for an entire league season, with Kinematics."""
    print(f"Fetching match list for Comp: {comp_id}, Season: {season_id}...")
    matches = sb.matches(competition_id=comp_id, season_id=season_id)
    match_ids = matches['match_id'].tolist()
    
    dataset = []
    tactical_keys = [
        'passes_attempted', 'pass_completion_pct', 'shots', 
        'goals', 'assists', 'carries', 'dribbles', 'fouls_won',
        'interceptions', 'tackles', 'clearances',
        'avg_action_velocity', 'max_action_velocity', 
        'explosive_bursts', 'total_action_distance', 'avg_deceleration'
    ]
    
    print(f"Processing {len(match_ids)} matches...")
    for m_id in tqdm(match_ids):
        try:
            events = sb.events(match_id=m_id)
            if 'player' not in events.columns or 'location' not in events.columns:
                continue
            
            # Kinematics Engine
            events['time_seconds'] = pd.to_timedelta(events['timestamp']).dt.total_seconds()
            valid_locs = events.dropna(subset=['location', 'player']).copy()
            valid_locs['x'] = valid_locs['location'].apply(lambda loc: loc[0])
            valid_locs['y'] = valid_locs['location'].apply(lambda loc: loc[1])
            
            valid_locs = valid_locs.sort_values(['player', 'time_seconds'])
            grouped = valid_locs.groupby('player')
            
            valid_locs['dt'] = grouped['time_seconds'].diff()
            valid_locs['dx'] = grouped['x'].diff()
            valid_locs['dy'] = grouped['y'].diff()
            valid_locs['dist'] = np.sqrt(valid_locs['dx']**2 + valid_locs['dy']**2)
            
            valid_locs['velocity'] = (valid_locs['dist'] / (valid_locs['dt'] + 1e-6)).clip(upper=10.0)
            valid_locs['dv'] = grouped['velocity'].diff()
            valid_locs['acceleration'] = (valid_locs['dv'] / (valid_locs['dt'] + 1e-6)).clip(lower=-6.0, upper=6.0)
            
            players_in_match = events['player'].dropna().unique()
            
            for player in players_in_match:
                player_events = events[events['player'] == player]
                player_kinematics = valid_locs[valid_locs['player'] == player]
                
                if len(player_events) < 15:
                    continue
                    
                loc_events = player_events[player_events['location'].notnull()]
                if len(loc_events) == 0:
                    continue
                    
                locations = np.array(loc_events['location'].tolist())
                heatmap, _, _ = np.histogram2d(
                    locations[:, 0], locations[:, 1], 
                    bins=[12, 8], range=[[0, 120], [0, 80]]
                )
                spatial_vector = heatmap.flatten()
                spatial_vector /= (spatial_vector.sum() + 1e-9) 
                
                tactical_stats = {key: 0.0 for key in tactical_keys}
                tactical_stats['passes_attempted'] = len(player_events[player_events['type'] == 'Pass'])
                if 'pass_outcome' in player_events.columns and tactical_stats['passes_attempted'] > 0:
                    completed = len(player_events[(player_events['type'] == 'Pass') & (player_events['pass_outcome'].isnull())])
                    tactical_stats['pass_completion_pct'] = completed / tactical_stats['passes_attempted']
                
                tactical_stats['shots'] = len(player_events[player_events['type'] == 'Shot'])
                tactical_stats['carries'] = len(player_events[player_events['type'] == 'Carry'])
                tactical_stats['dribbles'] = len(player_events[player_events['type'] == 'Dribble'])
                tactical_stats['interceptions'] = len(player_events[player_events['type'] == 'Interception'])
                
                if 'shot_outcome' in player_events.columns:
                    tactical_stats['goals'] = len(player_events[(player_events['type'] == 'Shot') & (player_events['shot_outcome'] == 'Goal')])
                if 'pass_goal_assist' in player_events.columns:
                    tactical_stats['assists'] = len(player_events[(player_events['type'] == 'Pass') & (player_events['pass_goal_assist'] == True)])
                    
                if len(player_kinematics) > 1:
                    tactical_stats['avg_action_velocity'] = player_kinematics['velocity'].mean()
                    tactical_stats['max_action_velocity'] = player_kinematics['velocity'].max()
                    tactical_stats['explosive_bursts'] = (player_kinematics['acceleration'] >= 3.0).sum()
                    tactical_stats['total_action_distance'] = player_kinematics['dist'].sum()
                    decel = player_kinematics[player_kinematics['acceleration'] < 0]['acceleration']
                    tactical_stats['avg_deceleration'] = decel.mean() if len(decel) > 0 else 0.0
                
                tactical_vector = [np.nan_to_num(tactical_stats[k]) for k in tactical_keys]
                
                dataset.append({
                    'match_id': m_id,
                    'player_name': player,
                    'spatial_vector': spatial_vector.tolist(),
                    'tactical_vector': tactical_vector
                })
        except Exception as e:
            continue
            
    df = pd.DataFrame(dataset)
    df.to_parquet(output_path)
    print(f"\nDataset built! Total Player-Match Vectors: {len(df)}")
    return df