# tennis-serve-pose-arm-vs-body

ETH semester project (3 ECTS), based on Williams et al., "Effects of variable pose input on neural network model performance for classifying basketball player activity".

**Question:** which traits does the serving arm alone (3 joints) carry, and which need the full body (17 joints)?

**Data:** 3D pose of 17 joints at 60 fps for 5,966 serves by 109 players (2024 US Open serve dataset), with metadata (player, gender, height, serve number, speed, match).

**Targets:** gender, player identity, serve number and serve speed.

**Method:** serving arm against all 17 joints, in raw, size-scaled and motion-only versions, with two models (a Random Forest on summary features and a small transformer on the raw positions). Models are always tested on players they have not seen (on matches they have not seen for identity), against baselines built from the metadata alone, with player-level bootstrap confidence intervals.

## Repository
- `notes/`: the hypothesis, the first speed check, how to run, and the folder structure (`FOLDER_STRUCTURE.txt` says what each file does)
- `splits/`: the train/test folds, in one place with its own README
- `src/`: shared code, `random_forest/`, `transformer/`, and `archive/` for earlier work

## Setup
1. `pip install -r requirements.txt`
2. Put the dataset in `tennis_serve_data/` (keypoints and `metadata.parquet`; see `tennis_serve_data/README.md`). It is not stored in the repo.
3. `notes/HOW_TO_RUN.txt` lists the commands in order. Results are written to `results/`, which is not part of the repo.
