Splits folder (all train/test folds live here)

splits.py                   makes the shared folds; the Random Forest and transformer scripts import it (python splits/splits.py rewrites the json)
shared_5fold_splits.json    the same folds as a file: test fold (0-4) of every serve, per task and repeat, -1 = not in the task

Shared 5-fold splits (3 repeats, seeds 0-2)
  player   whole players held out; gender, serve number, speed (all 5,966 serves)
  match    whole matches held out; identity, only the 46 players seen in 2+ matches (4,264 serves, 1,702 are -1);
           a test serve is scored only if its player is also in the training folds
  inner_split()   validation part cut from a TRAINING set (same grouping), for early stopping and tuning; not in the json
  Used by:   src/random_forest/rf_common.py (rf_classify.py, rf_speed.py), src/transformer/run_transformer.py
  In a script:   sys.path.insert(0, ROOT / "splits"); import splits
                 splits.folds("player", rep)  ->  5 (train_idx, test_idx) pairs over metadata rows

New experiments should use the shared 5-fold splits.
