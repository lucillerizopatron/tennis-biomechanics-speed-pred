# tennis-biomechanics-speed-pred

ETH semester project (3 ECTS), based on Williams et al., "Effects of variable pose input on neural network model performance for classifying basketball player activity".

**Question:** which traits does the serving arm alone (3 joints) carry, and which need the full body (17 joints)? We compare the two inputs on serve speed, serve number, gender and player identity, using 3D pose from the 2024 US Open serve dataset (5,966 serves, 109 players), splits by player (by match for identity) and metadata baselines.

*Work in progress.* See `notes/` for the hypothesis and the first speed check, and `notes/FOLDER_STRUCTURE.txt` for what each file does.

## Setup
1. `pip install -r requirements.txt`
2. Put the dataset in `tennis_serve_data/` (keypoints and `metadata.parquet`; see `tennis_serve_data/README.md`). It is not stored in the repo.
