"""
Tactical Engine - Streamlit Frontend (Offline Database Edition)
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from mplsoccer import Pitch
from scipy.spatial.distance import cosine

st.set_page_config(page_title="Deep Tactical Engine", layout="wide")

# --- CACHED DATABASE LOADING ---
# This reads our CSV once and keeps it in lightning-fast RAM
@st.cache_data
def load_database():
    try:
        df = pd.read_csv("tactical_database.csv")
        return df
    except FileNotFoundError:
        st.error("Database not found! Please run `python build_database.py` first.")
        st.stop()

df_db = load_database()

# Define the tactical columns based on our extraction keys
tactical_cols = [
    'passes_attempted', 'pass_completion_pct', 'shots', 
    'goals', 'assists', 'carries', 'dribbles', 'fouls_won'
]

clean_labels = {
    'passes_attempted': 'Passes',
    'pass_completion_pct': 'Pass Comp %',
    'shots': 'Shots',
    'goals': 'Goals',
    'assists': 'Assists',
    'carries': 'Carries',
    'dribbles': 'Dribbles',
    'fouls_won': 'Fouls Won'
}
clean_feat_names = [clean_labels.get(col, col) for col in tactical_cols]

# Extract spatial columns (spatial_0 to spatial_383)
spatial_cols = [f"spatial_{i}" for i in range(384)]
latent_cols = ["Dim_1_Playmaking", "Dim_2_Retention", "Dim_3_Finishing"]

# --- UI LAYOUT & DROPDOWNS ---
st.title("🧠 Deep Tactical Signature Profiling")
st.markdown("Compare players in a universal, fixed 3D Tactical Latent Space.")

players_available = sorted(df_db['Player'].unique())

st.sidebar.header("Player A Selection")
player_a = st.sidebar.selectbox("Player A Name", players_available, index=0)
seasons_a = sorted(df_db[df_db['Player'] == player_a]['Season'].unique())
season_a = st.sidebar.selectbox("Player A Season", seasons_a, index=0)

st.sidebar.header("Player B Selection")
# Default to a different player if possible, otherwise same player
default_b_index = 1 if len(players_available) > 1 else 0
player_b = st.sidebar.selectbox("Player B Name", players_available, index=default_b_index)
seasons_b = sorted(df_db[df_db['Player'] == player_b]['Season'].unique())
season_b = st.sidebar.selectbox("Player B Season", seasons_b, index=0)

# Extract subsets for the specific players and seasons
df_A = df_db[(df_db['Player'] == player_a) & (df_db['Season'] == season_a)]
df_B = df_db[(df_db['Player'] == player_b) & (df_db['Season'] == season_b)]

if len(df_A) == 0 or len(df_B) == 0:
    st.warning("No data available for this specific selection.")
    st.stop()

# --- CALCULATE AVERAGES ---
# Averages for the Bar Charts and Pitches
mean_spatial_A = df_A[spatial_cols].mean().values
mean_spatial_B = df_B[spatial_cols].mean().values

mean_tac_A = df_A[tactical_cols].mean().values
mean_tac_B = df_B[tactical_cols].mean().values

mean_latent_A = df_A[latent_cols].mean().values
mean_latent_B = df_B[latent_cols].mean().values

# --- PITCH VISUALIZATION BLOCK ---
col1, col2 = st.columns(2)

def draw_pitch_heatmap(spatial_vector, title):
    grid = spatial_vector.reshape(16, 24)
    fig, ax = plt.subplots(figsize=(8, 5))
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
    pitch.draw(ax=ax)
    ax.imshow(grid, extent=(0, 120, 80, 0), aspect='auto', cmap='magma', alpha=0.7)
    ax.set_title(title, color='white', fontsize=14)
    fig.patch.set_facecolor('#22312b')
    return fig

# Format names for cleaner display
name_a_clean = f"{player_a.split()[-2] if len(player_a.split()) > 1 else player_a} ({season_a})"
name_b_clean = f"{player_b.split()[-2] if len(player_b.split()) > 1 else player_b} ({season_b})"

with col1:
    st.subheader(f"Player A: {name_a_clean}")
    st.pyplot(draw_pitch_heatmap(mean_spatial_A, "Average Spatial Footprint"))
    
    chart_df_A = pd.DataFrame({"Metric": clean_feat_names, "Per Match Average": mean_tac_A})
    st.bar_chart(chart_df_A.set_index("Metric"), color="#00ffcc")

with col2:
    st.subheader(f"Player B: {name_b_clean}")
    st.pyplot(draw_pitch_heatmap(mean_spatial_B, "Average Spatial Footprint"))
    
    chart_df_B = pd.DataFrame({"Metric": clean_feat_names, "Per Match Average": mean_tac_B})
    st.bar_chart(chart_df_B.set_index("Metric"), color="#ff0066")

# --- LATENT SPACE 3D VISUALIZATION ---
st.markdown("---")
st.subheader("🌌 The Universal Tactical Latent Space")

st.info("""
**How to read this chart:**
* **Every dot** represents a single match.
* **The Space is Fixed:** This 3D universe was built across all players and seasons. The axes will never shift.
* Compare **Messi 11/12** vs **Messi 15/16** to see how his DNA drifted, or compare **Xavi** vs **Iniesta** to see role overlap.
""")

# Prepare data for 3D plot
plot_df_A = df_A.copy()
plot_df_A['Label'] = name_a_clean

plot_df_B = df_B.copy()
plot_df_B['Label'] = name_b_clean

plot_df_combined = pd.concat([plot_df_A, plot_df_B])

fig_3d = px.scatter_3d(
    plot_df_combined, 
    x="Dim_1_Playmaking", y="Dim_2_Retention", z="Dim_3_Finishing", 
    color="Label", opacity=0.8,
    color_discrete_sequence=["#00ffcc", "#ff0066"],
    hover_data=["Player", "Season"]
)

fig_3d.update_layout(
    scene=dict(
        bgcolor="#1e1e1e",
        xaxis=dict(title='Dim 1: Playmaking', backgroundcolor="#1e1e1e", gridcolor="gray"),
        yaxis=dict(title='Dim 2: Retention', backgroundcolor="#1e1e1e", gridcolor="gray"),
        zaxis=dict(title='Dim 3: Finishing', backgroundcolor="#1e1e1e", gridcolor="gray"),
    ),
    paper_bgcolor="#0e1117", 
    font=dict(color="white"),
    margin=dict(l=0, r=0, b=0, t=0),
    legend_title_text="Selection"
)
st.plotly_chart(fig_3d, use_container_width=True)

# --- SIMILARITY MATRIX ---
st.markdown("---")
st.subheader("🧬 Tactical DNA Similarity Matrix")

# Calculate Cosine Similarity (1 - distance). 
# We add a tiny epsilon (1e-9) to avoid division by zero if a vector is completely empty.
spatial_sim = 1 - cosine(mean_spatial_A + 1e-9, mean_spatial_B + 1e-9)
tactical_sim = 1 - cosine(mean_tac_A + 1e-9, mean_tac_B + 1e-9)
deep_sim = 1 - cosine(mean_latent_A + 1e-9, mean_latent_B + 1e-9)

sim_data = pd.DataFrame({
    "Similarity Dimension": ["Spatial (Where)", "Semantic (What)", "Deep Latent DNA (Overall Role)"],
    "Similarity Score (0-1)": [spatial_sim, tactical_sim, deep_sim]
})

st.table(sim_data.style.background_gradient(cmap='Purples', subset=['Similarity Score (0-1)']))