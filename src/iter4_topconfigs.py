"""Iteration 4 top-config submission files — FAST PATH.

Background: the original plan was full 5-fold CV re-runs of the 3 best
configs from submissions/iter4_search.csv. That proved infeasible: the VM
was replaced mid-task (all packages lost, reinstalled), and LightGBM
throughput in the current environment is ~1.1 sec/round with 3 training
jobs sharing 4 cores — a full 5-fold re-run would take 5+ hours for the
3 configs, missing the user's "10 submissions today" order.

This script instead trains ONE model per config on a stratified 90/10
split (seed 42): LightGBM with early stopping on the 10% holdout, then
isotonic calibration fit on holdout predictions (same calibration
discipline as iter4), then test predictions. ~8 min per config.

Configs (param overrides off LGBM_BASE), with search-time 5-fold CV
combined scores for reference (already verified in the original search;
not re-run here):
- smallfast: {"num_leaves": 31, "learning_rate": 0.05} -> 0.80496
- l2_3.0:    {"lambda_l2": 3.0}                      -> 0.80484
- l1_0.5:    {"lambda_l1": 0.5}                      -> 0.80469

Outputs per config <name>:
- submissions/iter4_<name>.csv (ID, Target) — submission file
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression

SEED = 42
TARGET = "liquidity_stress_next_30d"
ID_COL = "ID"
CAT_COLS = ["gender", "region", "smartphone", "segment", "earning_pattern"]
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "submissions")

BAL_COLS = [f"m{i}_daily_avg_bal" for i in range(6, 0, -1)]  # m6 oldest .. m1 newest
GROUPS = ["deposit", "withdraw", "received", "mm_send", "paybill",
          "merchantpay", "transfer_from_bank"]

LGBM_BASE = {
    "objective": "binary", "metric": "binary_logloss",
    "learning_rate": 0.04, "num_leaves": 63, "min_child_samples": 40,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "verbose": -1, "num_threads": 2, "seed": SEED,
}

# (name, param overrides off BASE, search-time 5-fold CV combined)
CONFIGS = [
    ("smallfast", {"num_leaves": 31, "learning_rate": 0.05}, 0.8049587938094367),
    ("l2_3.0", {"lambda_l2": 3.0}, 0.804840945553539),
    ("l1_0.5", {"lambda_l1": 0.5}, 0.8046894085441136),
]


def add_features(df):
    df = df.copy()
    b = df[BAL_COLS].values.astype(float)
    t = np.arange(6)  # 0=m6(oldest) .. 5=m1(newest)
    tc = t - t.mean()
    df["bal_slope"] = (b * tc).sum(axis=1) / (tc ** 2).sum()
    df["bal_std"] = b.std(axis=1)
    df["bal_min"] = b.min(axis=1)
    df["bal_max"] = b.max(axis=1)
    df["bal_mean"] = b.mean(axis=1)
    m1 = df["m1_daily_avg_bal"].values
    m6 = df["m6_daily_avg_bal"].values
    df["bal_m1_div_m6"] = m1 / np.maximum(m6, 1.0)
    df["bal_m1_minus_m6"] = m1 - m6
    mom = np.diff(b[:, ::-1], axis=1)  # m1-m2, m2-m3, ... (newest first diffs)
    df["bal_max_drop"] = (-mom).max(axis=1)
    df["bal_n_drops"] = (mom < 0).sum(axis=1)

    for g in GROUPS:
        tv = df[[f"m{i}_{g}_total_value" for i in range(1, 7)]].values.astype(float)
        recent = tv[:, 0] + tv[:, 1]
        old = tv[:, 4] + tv[:, 5]
        df[f"{g}_tv_sum"] = tv.sum(axis=1)
        df[f"{g}_tv_recent_div_old"] = recent / np.maximum(old, 1.0)
        df[f"{g}_tv_recent_minus_old"] = recent - old

    flows = []
    for i in range(1, 7):
        inflow = df[f"m{i}_deposit_total_value"] + df[f"m{i}_received_total_value"]
        outflow = df[f"m{i}_withdraw_total_value"] + df[f"m{i}_mm_send_total_value"]
        flows.append((inflow - outflow).values)
    f = np.stack(flows, axis=1).astype(float)  # m1..m6
    fr = f[:, ::-1]  # m6..m1 oldest first
    df["netflow_sum"] = fr.sum(axis=1)
    df["netflow_slope"] = (fr * tc).sum(axis=1) / (tc ** 2).sum()
    df["netflow_m1"] = f[:, 0]
    wsum = df["withdraw_tv_sum"].values
    dsum = df["deposit_tv_sum"].values
    df["withdraw_div_deposit"] = wsum / np.maximum(dsum, 1.0)
    return df


def combined(ll, auc):
    return 0.6 * (1 - ll) + 0.4 * auc


def run_single_split(X, y, Xte, params):
    """Single stratified 90/10 split: train on 90%, early-stop + calibrate
    isotonic on 10% holdout, predict test. Returns (holdout_ll, holdout_auc,
    holdout_combined, test_pred, best_iter)."""
    trn_idx, val_idx = train_test_split(
        np.arange(len(X)), test_size=0.10, random_state=SEED, stratify=y)
    dtr = lgb.Dataset(X.iloc[trn_idx], y[trn_idx], categorical_feature=CAT_COLS)
    dva = lgb.Dataset(X.iloc[val_idx], y[val_idx], categorical_feature=CAT_COLS,
                      reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    pv = m.predict(X.iloc[val_idx], num_iteration=m.best_iteration)
    pt = m.predict(Xte, num_iteration=m.best_iteration)
    iso = IsotonicRegression(out_of_bounds="clip")
    pv_cal = np.clip(iso.fit_transform(pv, y[val_idx]), 1e-6, 1 - 1e-6)
    test_pred = np.clip(iso.transform(np.clip(pt, 0, 1)), 1e-6, 1 - 1e-6)
    ll = log_loss(y[val_idx], pv_cal)
    auc = roc_auc_score(y[val_idx], pv_cal)
    return ll, auc, combined(ll, auc), test_pred, m.best_iteration


def main():
    tr = pd.read_csv(os.path.join(DATA, "Train.csv"))
    te = pd.read_csv(os.path.join(DATA, "Test.csv"))
    tr = add_features(tr)
    te = add_features(te)
    for c in CAT_COLS:
        tr[c] = tr[c].astype("category")
        te[c] = te[c].astype("category")
    feats = [c for c in tr.columns if c not in (ID_COL, TARGET)]
    print(f"n_features: {len(feats)}", flush=True)
    X, y, Xte = tr[feats], tr[TARGET].values, te[feats]
    os.makedirs(OUT, exist_ok=True)

    te_ids = te[ID_COL]
    for name, over, ref_score in CONFIGS:
        params = dict(LGBM_BASE)
        params.update(over)
        ll, auc, sc, test_pred, best_iter = run_single_split(X, y, Xte, params)
        print(f"config {name:12s} holdout LL={ll:.5f} AUC={auc:.5f} "
              f"comb={sc:.5f} (search 5-fold={ref_score:.5f}) "
              f"best_iter={best_iter}", flush=True)
        sub = pd.DataFrame({ID_COL: te_ids,
                            "Target": np.clip(test_pred, 1e-6, 1 - 1e-6)})
        sub_path = os.path.join(OUT, f"iter4_{name}.csv")
        sub.to_csv(sub_path, index=False)
        print(f"Saved {sub_path} (test mean {test_pred.mean():.4f})", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
