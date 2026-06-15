# ⚽ AI Tactical Scouting Engine

[![CI Pipeline](https://github.com/YOUR_USERNAME/ftbl-fingerprint/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/ftbl-fingerprint/actions)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR_APP_URL.streamlit.app/)

An intelligent scouting engine that uses deep non-linear manifold mapping (Variational Autoencoders) and high-resolution kinematic tracking to find statistically identical tactical replacements for elite football players.

![Portfolio Thumbnail](portfolio_thumbnail.png)

## 🧠 Methodology
Instead of relying on basic volume stats, this engine builds a **112+ Dimensional Fingerprint** for every player:
1. **Spatial Matrices:** Matches are converted into $24 \times 16$ finite-element heatmaps.
2. **Custom Kinematics Engine:** Calculates velocity, explosive bursts, and deceleration limits using temporal first-derivatives of event data.
3. **Latent Manifold Projection:** Folds the high-dimensional data into a continuous 16D latent space using a PyTorch VAE, isolating distinct player identities with a **>2.30x mathematical separation ratio**.

## 🚀 Quick Start
```bash
git clone [https://github.com/YOUR_USERNAME/ftbl-fingerprint.git](https://github.com/YOUR_USERNAME/ftbl-fingerprint.git)
cd ftbl-fingerprint
pip install -r requirements.txt
streamlit run app.py