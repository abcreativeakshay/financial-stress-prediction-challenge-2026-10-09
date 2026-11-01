"""Iteration 3: stacking (LGBM + XGB blend) + target encoding.

Changes vs iter2:
- Stacking: blend LGBM (native categoricals) + XGBoost (one-hot encoded
  categoricals) via OOF-weighted averaging. Weights chosen on OOF
  predictions to maximize the combined score (0.6*(1-LL)+0.4*AUC).
- XGBoost: one-hot for the 5 low-cardinality categoricals instead of
  native-categorical DMatrix (XGB 3.x native categoricals were underwhelming).
- Target encoding: out-of-fold target means (alpha=20 smoothing) for
  region/segment/earning_pattern, added as features for LGBM.
- Keeps iter2 engineered features + per-fold isotonic calibration.

All CV honest: OOF predictions from 5-fold stratified split (seed 42);
target encodings computed inside each fold from train folds only;
calibration fit on validation predictions only. Best blend (by OOF combined
score) -> submissions/iter3.csv
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
TE_COLS = ["region", "segment", "earning_pattern"]
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "submissions")

BAL_COLS = [f"m{i}_daily_avg_bal" for i in range(6, 0, -1)]
GROUPS = ["deposit", "withdraw", "received", "mm_send", "paybill",
          "merchantpay", "transfer_from_bank"]

LGBM_PARAMS = {
    "objective": "binary", "metric": "binary_logloss",
    "learning_rate": 0.03, "num_leaves": 127, "min_child_samples": 25,
    "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1,
    "lambda_l2": 2.0, "verbose": -1, "num_threads": 4, "seed": SEED,
}
XGB_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "logloss",
    "learning_rate": 0.04, "max_depth": 6, "min_child_weight": 30,
    "subsample": 0.8, "colsample_bytree": 0.7, "reg_lambda": 2.0,
    "tree_method": "hist", "nthread": 4, "seed": SEED,
}


def add_features(df):
    df = df.copy()
    b = df[BAL_COLS].values.astype(float)
    t = np.arange(6)
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
    mom = np.diff(b[:, ::-1], axis=1)
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
    f = np.stack(flows, axis=1).astype(float)
    fr = f[:, ::-1]
    df["netflow_sum"] = fr.sum(axis=1)
    df["netflow_slope"] = (fr * tc).sum(axis=1) / (tc ** 2).sum()
    df["netflow_m1"] = f[:, 0]
    wsum = df["withdraw_tv_sum"].values
    dsum = df["deposit_tv_sum"].values
    df["withdraw_div_deposit"] = wsum / np.maximum(dsum, 1.0)
    return df


def combined(ll, auc):
    return 0.6 * (1 - ll) + 0.4 * auc


def onehot_frame(dfn):
    """One-hot the 5 categoricals; pass numerics through."""
    return pd.get_dummies(dfn, columns=CAT_COLS, dtype=np.float32)


def target_encode(train_fold, val_fold, test, y_trn, alpha=20.0):
    """Fold-honest smoothed target means; adds te_* columns. Returns new frames."""
    glob = y_trn.mean()
    for c in TE_COLS:
        agg = pd.DataFrame({"y": y_trn, "c": train_fold[c].values}).groupby("c")["y"].agg(["sum", "count"])
        enc = (agg["sum"] + glob * alpha) / (agg["count"] + alpha)
        for fr in (train_fold, val_fold, test):
            fr[f"te_{c}"] = fr[c].astype(object).map(enc).fillna(glob).values
    return train_fold, val_fold, test


def run_lgbm_fold(trn_df, val_df, y_trn, y_val, te_df, feats):
    dtr = lgb.Dataset(trn_df[feats], y_trn, categorical_feature=CAT_COLS)
    dva = lgb.Dataset(val_df[feats], y_val, categorical_feature=CAT_COLS, reference=dtr)
    m = lgb.train(LGBM_PARAMS, dtr, num_boost_round=5000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    pv = m.predict(val_df[feats], num_iteration=m.best_iteration)
    pt = m.predict(te_df[feats], num_iteration=m.best_iteration)
    return pv, pt, m.best_iteration


def run_xgb_fold(trn_df, val_df, y_trn, y_val, te_df):
    oh = onehot_frame
    Xtr, Xva, Xte = oh(trn_df.drop(columns=[ID_COL, TARGET])), \
        oh(val_df.drop(columns=[ID_COL, TARGET])), oh(te_df.drop(columns=[ID_COL]))
    # align columns (should be identical, but be safe)
    Xva = Xva.reindex(columns=Xtr.columns, fill_value=0)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
    dtr = xgb.DMatrix(Xtr, label=y_trn)
    dva = xgb.DMatrix(Xva, label=y_val)
    dte = xgb.DMatrix(Xte)
    m = xgb.train(XGB_PARAMS, dtr, num_boost_round=5000,
                  evals=[(dva, "val")], early_stopping_rounds=200, verbose_eval=False)
    return m.predict(dva), m.predict(dte), m.best_iteration


def report(tag, y, oof):
    o = np.clip(oof, 1e-6, 1 - 1e-6)
    ll, auc = log_loss(y, o), roc_auc_score(y, o)
    print(f"{tag:30s} LL={ll:.5f} AUC={auc:.5f} comb={combined(ll, auc):.5f}", flush=True)
    return ll, auc


def main():
    tr = pd.read_csv(os.path.join(DATA, "Train.csv"))
    te = pd.read_csv(os.path.join(DATA, "Test.csv"))
    tr = add_features(tr)
    te = add_features(te)
    for c in CAT_COLS:
        tr[c] = tr[c].astype("category")
        te[c] = te[c].astype("category")

    feats = [c for c in tr.columns if c not in (ID_COL, TARGET)]
    feats_te = feats + [f"te_{c}" for c in TE_COLS]
    print(f"n_features base={len(feats)} (+3 target-encoded for LGBM)", flush=True)

    y = tr[TARGET].values
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof_lgbm = np.zeros(len(tr))
    oof_xgb = np.zeros(len(tr))
    oof_cal_l = np.zeros(len(tr))
    oof_cal_x = np.zeros(len(tr))
    t_lgbm = np.zeros(len(te))
    t_xgb = np.zeros(len(te))

    for fold, (ti, vi) in enumerate(skf.split(tr, y)):
        trn, val = tr.iloc[ti].copy().reset_index(drop=True), \
            tr.iloc[vi].copy().reset_index(drop=True)
        tst = te.copy().reset_index(drop=True)
        trn, val, tst = target_encode(trn, val, tst, y[ti])

        pv, pt, it = run_lgbm_fold(trn, val, y[ti], y[vi], tst, feats_te)
        oof_lgbm[vi] = pv
        t_lgbm += pt / N_SPLITS

        pvx, ptx, itx = run_xgb_fold(trn, val, y[ti], y[vi], tst)
        oof_xgb[vi] = pvx
        t_xgb += ptx / N_SPLITS

        print(f"fold {fold}: lgbm_iter={it} xgb_iter={itx}", flush=True)

    print("\n--- OOF (raw) ---", flush=True)
    report("lgbm+feats+TE", y, oof_lgbm)
    report("xgb+onehot+feats", y, oof_xgb)

    # OOF-calibrate each, then grid-search blend weights on calibrated OOF
    iso_l = IsotonicRegression(out_of_bounds="clip")
    iso_x = IsotonicRegression(out_of_bounds="clip")
    oof_cal_l[:] = np.clip(iso_l.fit_transform(np.clip(oof_lgbm, 0, 1), y), 1e-6, 1 - 1e-6)
    oof_cal_x[:] = np.clip(iso_x.fit_transform(np.clip(oof_xgb, 0, 1), y), 1e-6, 1 - 1e-6)
    print("--- OOF (isotonic-calibrated) ---", flush=True)
    report("lgbm cal", y, oof_cal_l)
    report("xgb cal", y, oof_cal_x)

    best_w, best_sc = 0.5, -1
    for w in np.arange(0.0, 1.01, 0.05):
        b = w * oof_cal_l + (1 - w) * oof_cal_x
        o = np.clip(b, 1e-6, 1 - 1e-6)
        sc = combined(log_loss(y, o), roc_auc_score(y, o))
        if sc > best_sc:
            best_w, best_sc = w, sc
    print(f"\nbest blend w_lgbm={best_w:.2f} OOF combined={best_sc:.5f}", flush=True)

    # also try calibrating the blend itself
    blend = best_w * oof_cal_l + (1 - best_w) * oof_cal_x
    iso_b = IsotonicRegression(out_of_bounds="clip")
    blend_cal = np.clip(iso_b.fit_transform(np.clip(blend, 0, 1), y), 1e-6, 1 - 1e-6)
    ll, auc = report("blend cal", y, blend_cal)

    # test predictions through the same calibrations
    t_cal_l = np.clip(iso_l.transform(np.clip(t_lgbm, 0, 1)), 1e-6, 1 - 1e-6)
    t_cal_x = np.clip(iso_x.transform(np.clip(t_xgb, 0, 1)), 1e-6, 1 - 1e-6)
    t_blend = best_w * t_cal_l + (1 - best_w) * t_cal_x
    t_final = np.clip(iso_b.transform(np.clip(t_blend, 0, 1)), 1e-6, 1 - 1e-6)

    os.makedirs(OUT, exist_ok=True)
    sub = pd.DataFrame({ID_COL: te[ID_COL], "Target": t_final})
    sub.to_csv(os.path.join(OUT, "iter3.csv"), index=False)
    pd.DataFrame({"ID": tr[ID_COL], "oof_pred": blend_cal, TARGET: y}).to_csv(
        os.path.join(OUT, "oof_iter3.csv"), index=False)
    print(f"\nSaved submissions/iter3.csv (test mean {t_final.mean():.4f}, "
          f"min {t_final.min():.4f}, max {t_final.max():.4f})")


if __name__ == "__main__":
    main()
