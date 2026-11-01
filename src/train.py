"""Baseline: LightGBM 5-fold stratified CV for liquidity stress prediction.

Target: liquidity_stress_next_30d (binary, 15% positive).
Metric on Zindi: 0.6 * LogLoss + 0.4 * ROC-AUC (higher better) --
we optimize logloss with early stopping and report both logloss and AUC.
Test predictions = mean of the 5 fold models.
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score

SEED = 42
N_SPLITS = 5
TARGET = "liquidity_stress_next_30d"
ID_COL = "ID"
CAT_COLS = ["gender", "region", "smartphone", "segment", "earning_pattern"]
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "submissions")

PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.04,
    "num_leaves": 63,
    "min_child_samples": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "num_threads": 4,
    "seed": SEED,
}


def load():
    tr = pd.read_csv(os.path.join(DATA, "Train.csv"))
    te = pd.read_csv(os.path.join(DATA, "Test.csv"))
    for c in CAT_COLS:
        tr[c] = tr[c].astype("category")
        te[c] = te[c].astype("category")
    feats = [c for c in tr.columns if c not in (ID_COL, TARGET)]
    return tr, te, feats


def main():
    tr, te, feats = load()
    X = tr[feats]
    y = tr[TARGET].values
    Xte = te[feats]

    oof = np.zeros(len(tr))
    test_pred = np.zeros(len(te))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    importances = []

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
        dtr = lgb.Dataset(X.iloc[trn_idx], y[trn_idx], categorical_feature=CAT_COLS)
        dva = lgb.Dataset(X.iloc[val_idx], y[val_idx], categorical_feature=CAT_COLS,
                          reference=dtr)
        model = lgb.train(
            PARAMS, dtr, num_boost_round=5000, valid_sets=[dva],
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        oof[val_idx] = model.predict(X.iloc[val_idx], num_iteration=model.best_iteration)
        test_pred += model.predict(Xte, num_iteration=model.best_iteration) / N_SPLITS
        importances.append(
            pd.Series(model.feature_importance(importance_type="gain"), index=feats)
        )
        print(f"fold {fold}: best_iter={model.best_iteration}", flush=True)

    cv_ll = log_loss(y, oof)
    cv_auc = roc_auc_score(y, oof)
    print(f"\nCV LogLoss: {cv_ll:.5f}")
    print(f"CV ROC-AUC: {cv_auc:.5f}")
    print(f"CV combined (0.6*(1-LL)+0.4*AUC): {0.6*(1-cv_ll)+0.4*cv_auc:.5f}")

    imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    print("\nTop 15 features by gain:")
    print(imp.head(15).to_string())

    os.makedirs(OUT, exist_ok=True)
    sub = pd.DataFrame({ID_COL: te[ID_COL], "Target": np.clip(test_pred, 1e-6, 1 - 1e-6)})
    sub.to_csv(os.path.join(OUT, "baseline_lgbm.csv"), index=False)
    pd.DataFrame({"ID": tr[ID_COL], "oof_pred": oof, TARGET: y}).to_csv(
        os.path.join(OUT, "oof_baseline_lgbm.csv"), index=False)
    print(f"\nSaved submissions/baseline_lgbm.csv ({len(sub)} rows)")
    print(f"Test pred mean: {test_pred.mean():.4f} (train positive rate: {y.mean():.4f})")


if __name__ == "__main__":
    main()
