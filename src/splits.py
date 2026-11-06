"""Shared train/test splits for every model (Random Forest and transformer).

Two tasks, both 5-fold, repeated with different seeds:
  player  : whole players held out (gender, serve number, serve speed)
  match   : whole matches held out (player identity); only players seen in 2+ matches

Rows follow the order of serve_data.load_metadata(), so an index here is the same
row in the metadata and in the keypoint list. Rows not part of a task get fold -1.
`python splits.py` writes splits/folds.json so the folds can be checked or shared.
"""
import json

import numpy as np
from sklearn.model_selection import GroupKFold

import serve_data as sd

N_FOLDS, N_REPS = 5, 3
OUT = sd.ROOT / "splits" / "folds.json"


def _meta():
    return sd.load_metadata().reset_index(drop=True)


def _task_rows(meta, task):
    """Row indices in the task and the column whose groups are held out."""
    if task == "player":
        return np.arange(len(meta)), "server"
    if task == "match":
        n_matches = meta.groupby("server").match_id.nunique()
        return np.where(meta.server.isin(n_matches[n_matches >= 2].index))[0], "match_id"
    raise ValueError(f"unknown task {task!r} (use 'player' or 'match')")


def fold_ids(task, rep=0, meta=None):
    """Array with the test fold (0-4) of every serve, -1 if the serve is not in the task."""
    meta = _meta() if meta is None else meta
    rows, col = _task_rows(meta, task)
    groups = meta[col].values[rows]
    ups = np.unique(groups)
    order = dict(zip(ups, np.random.RandomState(rep).permutation(len(ups))))
    gid = np.array([order[g] for g in groups])
    fold = np.full(len(meta), -1)
    for k, (_, te) in enumerate(GroupKFold(N_FOLDS).split(rows, groups=gid)):
        fold[rows[te]] = k
    return fold


def folds(task, rep=0, meta=None):
    """List of (train_idx, test_idx) over metadata row indices."""
    f = fold_ids(task, rep, meta)
    return [(np.where((f >= 0) & (f != k))[0], np.where(f == k)[0]) for k in range(N_FOLDS)]


def inner_split(train_idx, task, meta=None, frac=0.15, seed=0):
    """Split a TRAINING set into (fit, validation) by the same groups, for early stopping
    or tuning. Never touches the test fold."""
    meta = _meta() if meta is None else meta
    col = "server" if task == "player" else "match_id"
    groups = meta[col].values[train_idx]
    ups = np.unique(groups)
    val = np.random.RandomState(seed).choice(ups, max(1, int(frac * len(ups))), replace=False)
    is_val = np.isin(groups, val)
    return train_idx[~is_val], train_idx[is_val]


def seen_in_train(meta, train_idx, test_idx):
    """Identity only: test serves whose player also appears in training."""
    return np.isin(meta.server.values[test_idx], np.unique(meta.server.values[train_idx]))


if __name__ == "__main__":
    meta = _meta()
    out = {t: {str(r): fold_ids(t, r, meta).tolist() for r in range(N_REPS)} for t in ("player", "match")}
    out["_info"] = {"n_serves": len(meta), "n_folds": N_FOLDS, "n_reps": N_REPS,
                    "note": "fold of each serve in metadata order; -1 = not in the task"}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out))
    for t in ("player", "match"):
        f = np.array(out[t]["0"])
        print(t, "rep 0: serves per fold", [int((f == k).sum()) for k in range(N_FOLDS)], "excluded", int((f < 0).sum()))
