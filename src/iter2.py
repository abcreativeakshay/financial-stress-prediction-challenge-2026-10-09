"""Iteration 2: feature engineering + calibration + model comparison.

Features added on top of raw 183:
- Balance trajectory (m1..m6 daily_avg_bal): slope, std, min, max, mean,
  m1/m6 ratio, m1-m6 diff, max month-over-month drop
- Per activity group (deposit, withdraw, received, mm_send, paybill,
  merchantpay, transfer_from_bank): total_value sum, recent(m1+m2) vs
  old(m5+m6) ratio
- Net flow per month = (deposit+received) - (withdraw+mm_send): sum, slope
- withdraw/deposit total ratio

Models (5-fold stratified CV, seed 42):
- LGBM baseline params + new feats
- LGBM tuned params + new feats
- XGBoost (hist, native categorical) + new feats
Each with per-fold isotonic calibration fit on validation predictions
(honest: calibration never sees the fold's training data).
Best calibrated combined score -> submissions/iter2.csv
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
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
LGBM_TUNED = {
    "objective": "binary", "metric": "binary_logloss",
    "learning_rate": 0.03, "num_leaves": 127, "min_child_samples": 25,
    "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1,
    "lambda_l2": 2.0, "verbose": -1, "num_threads": 4, "seed": SEED,
}
XGB_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "logloss",
    "learning_rate": 0.04, "max_depth": 6, "min_child_weight": 30,
    "subsample": 0.8, "colsample_bytree": 0.7, "reg_lambda": 2.0,
    "tree_method": "hist", "enable_categorical": True,
    "nthread": 4, "seed": SEED,
}


def add_features(df):
    df = df.copy()
    b = df[BAL_COLS].values.astype(float)
    t = np.arange(6)  # 0=m6(oldest) .. 5=m1(newest)
    # slope via least squares
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
    df["bal_max_drop"] = (-mom).max(axis=1)  # largest single-month drop
    df["bal_n_drops"] = (mom < 0).sum(axis=1)

    for g in GROUPS:
        tv = df[[f"m{i}_{g}_total_value" for i in range(1, 7)]].values.astype(float)
        recent = tv[:, 0] + tv[:, 1]
        old = tv[:, 4] + tv[:, 5]
        df[f"{g}_tv_sum"] = tv.sum(axis=1)
        df[f"{g}_tv_recent_div_old"] = recent / np.maximum(old, 1.0)
        df[f"{g}_tv_recent_minus_old"] = recent - old

    # net flow per month: inflows - outflows
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


def run_lgbm(X, y, Xte, feats, params, name):
    oof = np.zeros(len(X))
    oof_cal = np.zeros(len(X))
    test_pred = np.zeros(len(Xte))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (trn, val) in enumerate(skf.split(X, y)):
        dtr = lgb.Dataset(X.iloc[trn], y[trn], categorical_feature=CAT_COLS)
        dva = lgb.Dataset(X.iloc[val], y[val], categorical_feature=CAT_COLS,
                          reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        pv = m.predict(X.iloc[val], num_iteration=m.best_iteration)
        pt = m.predict(Xte, num_iteration=m.best_iteration)
        oof[val] = pv
        iso = IsotonicRegression(out_of_bounds="clip")
        oof_cal[val] = np.clip(iso.fit_transform(pv, y[val]), 1e-6, 1 - 1e-6)
        test_pred += np.clip(iso.transform(np.clip(pt, 0, 1)), 1e-6, 1 - 1e-6) / N_SPLITS
        print(f"  {name} fold {fold}: iter={m.best_iteration}", flush=True)
    return oof, oof_cal, test_pred


def run_xgb(X, y, Xte, feats, name):
    oof = np.zeros(len(X))
    oof_cal = np.zeros(len(X))
    test_pred = np.zeros(len(Xte))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (trn, val) in enumerate(skf.split(X, y)):
        dtr = xgb.DMatrix(X.iloc[trn], label=y[trn], enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[val], label=y[val], enable_categorical=True)
        dte = xgb.DMatrix(Xte, enable_categorical=True)
        m = xgb.train(XGB_PARAMS, dtr, num_boost_round=5000,
                      evals=[(dva, "val")], early_stopping_rounds=200,
                      verbose_eval=False)
        pv = m.predict(dva)
        pt = m.predict(dte)
        oof[val] = pv
        iso = IsotonicRegression(out_of_bounds="clip")
        oof_cal[val] = np.clip(iso.fit_transform(pv, y[val]), 1e-6, 1 - 1e-6)
        test_pred += np.clip(iso.transform(np.clip(pt, 0, 1)), 1e-6, 1 - 1e-6) / N_SPLITS
        print(f"  {name} fold {fold}: iter={m.best_iteration}", flush=True)
    return oof, oof_cal, test_pred


def report(tag, y, oof, oof_cal):
    ll, auc = log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6)), roc_auc_score(y, oof)
    llc, aucc = log_loss(y, oof_cal), roc_auc_score(y, oof_cal)
    print(f"{tag:28s} raw: LL={ll:.5f} AUC={auc:.5f} comb={combined(ll,auc):.5f} | "
          f"cal: LL={llc:.5f} AUC={aucc:.5f} comb={combined(llc,aucc):.5f}", flush=True)
    return combined(llc, aucc)


def main():
    tr = pd.read_csv(os.path.join(DATA, "Train.csv"))
    te = pd.read_csv(os.path.join(DATA, "Test.csv"))
    tr = add_features(tr)
    te = add_features(te)
    for c in CAT_COLS:
        tr[c] = tr[c].astype("category")
        te[c] = te[c].astype("category")
    feats = [c for c in tr.columns if c not in (ID_COL, TARGET)]
    print(f"n_features: {len(feats)} (raw 183 + {len(feats)-183} engineered)", flush=True)
    X, y, Xte = tr[feats], tr[TARGET].values, te[feats]

    results = {}
    oof, oof_cal, tp = run_lgbm(X, y, Xte, feats, LGBM_BASE, "lgbm_base")
    results["lgbm_base"] = (report("lgbm_base+feats", y, oof, oof_cal), tp, oof_cal)
    oof, oof_cal, tp = run_lgbm(X, y, Xte, feats, LGBM_TUNED, "lgbm_tuned")
    results["lgbm_tuned"] = (report("lgbm_tuned+feats", y, oof, oof_cal), tp, oof_cal)
    oof, oof_cal, tp = run_xgb(X, y, Xte, feats, "xgb")
    results["xgb"] = (report("xgb+feats", y, oof, oof_cal), tp, oof_cal)

    best = max(results, key=lambda k: results[k][0])
    score, test_pred, oof_cal = results[best]
    print(f"\nBEST: {best} calibrated combined={score:.5f}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    sub = pd.DataFrame({ID_COL: te[ID_COL],
                        "Target": np.clip(test_pred, 1e-6, 1 - 1e-6)})
    sub.to_csv(os.path.join(OUT, "iter2.csv"), index=False)
    pd.DataFrame({"ID": tr[ID_COL], "oof_pred_cal": oof_cal, TARGET: y}).to_csv(
        os.path.join(OUT, "oof_iter2.csv"), index=False)
    print(f"Saved submissions/iter2.csv via {best} "
          f"(test mean {test_pred.mean():.4f})")


if __name__ == "__main__":
    main()
