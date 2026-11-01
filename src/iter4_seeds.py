"""Seed variants of the iteration-4 winner (num_leaves=31).

Same config as iter4's best: LGBM_BASE + {"num_leaves": 31}, 216 features
from add_features() (copied verbatim from iter4.py), 5-fold stratified CV,
per-fold isotonic calibration, combined = 0.6*(1-LogLoss) + 0.4*ROC-AUC.

Seeds 43, 44, 45: both the CV split seed AND the LightGBM "seed" param are
changed, so each variant is a genuinely different model for ensembling /
multi-submission purposes.
"""
import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression

N_SPLITS = 5
SEEDS = [int(s) for s in sys.argv[1:]] or [43, 44, 45]
TARGET = "liquidity_stress_next_30d"
ID_COL = "ID"
CAT_COLS = ["gender", "region", "smartphone", "segment", "earning_pattern"]
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "submissions")

BAL_COLS = [f"m{i}_daily_avg_bal" for i in range(6, 0, -1)]  # m6 oldest .. m1 newest
GROUPS = ["deposit", "withdraw", "received", "mm_send", "paybill",
          "merchantpay", "transfer_from_bank"]


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


def run_seed(X, y, Xte, seed):
    """5-fold stratified CV (split seed=seed) + per-fold isotonic calibration.
    LightGBM params = LGBM_BASE + num_leaves=31 + seed=seed."""
    params = {
        "objective": "binary", "metric": "binary_logloss",
        "learning_rate": 0.04, "num_leaves": 31, "min_child_samples": 40,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
        "lambda_l2": 1.0, "verbose": -1, "num_threads": 2, "seed": seed,
    }
    oof_cal = np.zeros(len(X))
    test_pred = np.zeros(len(Xte))
    iters = []
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for fold, (trn, val) in enumerate(skf.split(X, y)):
        dtr = lgb.Dataset(X.iloc[trn], y[trn], categorical_feature=CAT_COLS)
        dva = lgb.Dataset(X.iloc[val], y[val], categorical_feature=CAT_COLS,
                          reference=dtr)
        # NB: rounds capped at 1500 (seed-42 best was 469; cap only binds in
        # pathological no-early-stopping cases and keeps runtime bounded)
        m = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        pv = m.predict(X.iloc[val], num_iteration=m.best_iteration)
        pt = m.predict(Xte, num_iteration=m.best_iteration)
        iso = IsotonicRegression(out_of_bounds="clip")
        oof_cal[val] = np.clip(iso.fit_transform(pv, y[val]), 1e-6, 1 - 1e-6)
        test_pred += np.clip(iso.transform(np.clip(pt, 0, 1)), 1e-6, 1 - 1e-6) / N_SPLITS
        iters.append(m.best_iteration)
        print(f"  seed {seed} fold {fold}: best_iter={m.best_iteration}", flush=True)
    return oof_cal, test_pred, float(np.mean(iters))


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
    test_ids = te[ID_COL].values
    assert (test_ids == pd.read_csv(os.path.join(DATA, "Test.csv"))[ID_COL].values).all()

    os.makedirs(OUT, exist_ok=True)
    results = []
    for seed in SEEDS:
        print(f"=== seed {seed} ===", flush=True)
        oof_cal, test_pred, mean_iter = run_seed(X, y, Xte, seed)
        ll = log_loss(y, oof_cal)
        auc = roc_auc_score(y, oof_cal)
        sc = combined(ll, auc)
        results.append((seed, ll, auc, sc, mean_iter))
        print(f"seed {seed}: LL={ll:.5f} AUC={auc:.5f} comb={sc:.5f} "
              f"mean_iter={mean_iter:.0f} test_mean={test_pred.mean():.4f}", flush=True)

        sub = pd.DataFrame({ID_COL: test_ids,
                            "Target": np.clip(test_pred, 1e-6, 1 - 1e-6)})
        assert len(sub) == 30000 and sub["Target"].notna().all()
        sub.to_csv(os.path.join(OUT, f"iter4_seed{seed}.csv"), index=False)
        pd.DataFrame({"ID": tr[ID_COL], "oof_pred_cal": oof_cal, TARGET: y}).to_csv(
            os.path.join(OUT, f"oof_iter4_seed{seed}.csv"), index=False)

    print("\n=== SUMMARY ===", flush=True)
    for seed, ll, auc, sc, mi in results:
        print(f"seed {seed}: LL={ll:.5f} AUC={auc:.5f} combined={sc:.5f}", flush=True)


if __name__ == "__main__":
    main()
