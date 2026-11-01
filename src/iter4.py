"""Iteration 4: hyperparameter tuning on iter2's winning architecture.

Angle (different from iter3's stacking, which hurt: 0.80246 < 0.80415):
systematic search around LGBM_BASE -- iter2's winner (CV combined 0.80415
with LL=0.25788, AUC=0.89720). iter2 only tried 2 hand-picked param sets;
the hand-tuned one was worse, so there is likely headroom around BASE.

Kept identical to iter2 for comparability:
- Same 40 engineered features (add_features copied verbatim)
- 5-fold stratified CV, seed 42
- Per-fold isotonic calibration (fit on validation preds only)
- Metric: combined = 0.6*(1-LogLoss) + 0.4*ROC-AUC on calibrated OOF

Search: 12 configs (BASE reproduction + 11 variants, single-param moves
off BASE plus 2 combos). Best calibrated combined -> submissions/iter4.csv,
oof_iter4.csv. Per-config scores appended to submissions/iter4_search.csv
as the search runs (crash-safe).
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression

SEED = 42
N_SPLITS = 5
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
    "lambda_l2": 1.0, "verbose": -1, "num_threads": 4, "seed": SEED,
}

# (name, param overrides off BASE)
SEARCH = [
    ("base", {}),
    ("lr_0.03", {"learning_rate": 0.03}),
    ("lr_0.06", {"learning_rate": 0.06}),
    ("leaves_31", {"num_leaves": 31}),
    ("minchild_80", {"min_child_samples": 80}),
    ("featfrac_0.7", {"feature_fraction": 0.7}),
    ("featfrac_0.9", {"feature_fraction": 0.9}),
    ("l2_3.0", {"lambda_l2": 3.0}),
    ("l1_0.5", {"lambda_l1": 0.5}),
    ("nobag", {"bagging_fraction": 1.0, "bagging_freq": 0}),
    ("reg_combo", {"min_child_samples": 80, "feature_fraction": 0.7,
                   "lambda_l2": 2.0}),
    ("smallfast", {"num_leaves": 31, "learning_rate": 0.05}),
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


def run_config(X, y, Xte, params):
    """5-fold CV with per-fold isotonic calibration. Returns oof_cal, test_pred, mean_iter."""
    oof_cal = np.zeros(len(X))
    test_pred = np.zeros(len(Xte))
    iters = []
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (trn, val) in enumerate(skf.split(X, y)):
        dtr = lgb.Dataset(X.iloc[trn], y[trn], categorical_feature=CAT_COLS)
        dva = lgb.Dataset(X.iloc[val], y[val], categorical_feature=CAT_COLS,
                          reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        pv = m.predict(X.iloc[val], num_iteration=m.best_iteration)
        pt = m.predict(Xte, num_iteration=m.best_iteration)
        iso = IsotonicRegression(out_of_bounds="clip")
        oof_cal[val] = np.clip(iso.fit_transform(pv, y[val]), 1e-6, 1 - 1e-6)
        test_pred += np.clip(iso.transform(np.clip(pt, 0, 1)), 1e-6, 1 - 1e-6) / N_SPLITS
        iters.append(m.best_iteration)
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

    os.makedirs(OUT, exist_ok=True)
    search_path = os.path.join(OUT, "iter4_search.csv")
    if os.path.exists(search_path):
        os.remove(search_path)

    best = {"name": None, "score": -1.0}
    store = {}  # name -> (oof_cal, test_pred)
    for name, over in SEARCH:
        params = dict(LGBM_BASE)
        params.update(over)
        oof_cal, test_pred, mean_iter = run_config(X, y, Xte, params)
        ll = log_loss(y, oof_cal)
        auc = roc_auc_score(y, oof_cal)
        sc = combined(ll, auc)
        store[name] = (oof_cal, test_pred)
        print(f"config {name:12s} LL={ll:.5f} AUC={auc:.5f} comb={sc:.5f} "
              f"mean_iter={mean_iter:.0f}", flush=True)
        pd.DataFrame([{"config": name, "ll": ll, "auc": auc,
                       "combined": sc, "mean_iter": mean_iter,
                       "overrides": str(over)}]).to_csv(
            search_path, mode="a", header=not os.path.exists(search_path),
            index=False)
        if sc > best["score"]:
            best = {"name": name, "score": sc, "ll": ll, "auc": auc}

    print(f"\nBEST: {best['name']} LL={best['ll']:.5f} AUC={best['auc']:.5f} "
          f"comb={best['score']:.5f}  (iter2 baseline: 0.80415)", flush=True)

    oof_cal, test_pred = store[best["name"]]
    sub = pd.DataFrame({ID_COL: te[ID_COL],
                        "Target": np.clip(test_pred, 1e-6, 1 - 1e-6)})
    sub.to_csv(os.path.join(OUT, "iter4.csv"), index=False)
    pd.DataFrame({"ID": tr[ID_COL], "oof_pred_cal": oof_cal, TARGET: y}).to_csv(
        os.path.join(OUT, "oof_iter4.csv"), index=False)
    print(f"Saved submissions/iter4.csv via config '{best['name']}' "
          f"(test mean {test_pred.mean():.4f})", flush=True)


if __name__ == "__main__":
    main()
