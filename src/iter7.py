"""Calibration bake-off on RAW pooled OOF (rerun), then full-data retrain.

Why a rerun: the bake-off on submissions/oof_iter6.csv was degenerate --
that OOF is already per-fold-isotonic-calibrated, so every calibrator
(isotonic/beta/temperature) converged to the identity map (combined 0.81140
for all four). The honest pooled-OOF protocol (Eskandar 2023) needs the
calibrator fit in RAW model-output space. So:

Phase 1: 5-fold stratified CV (seed 42) on iter6's final 266 features with
         KITCHEN_SINK params, saving RAW (uncalibrated) OOF predictions.
Phase 2: bake-off on raw pooled OOF -- none vs pooled isotonic vs beta
         (Kull et al. 2017) vs temperature scaling -- by combined metric.
Phase 3: retrain single LightGBM on FULL 40k rows (645 rounds = CV mean
         best_iteration of kitchen_sink), apply winning pooled calibrator
         to raw test predictions, clip, save submissions/iter7.csv.

Feature construction is imported verbatim from iter6.py; final feature list
= base_cols + kept_new from submissions/iter6_state.json (266 features).
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iter6 import (add_features_base, add_features_expanded,
                   add_target_encoding, KITCHEN_SINK, SEED, N_SPLITS,
                   TARGET, ID_COL, CAT_COLS, DATA, OUT, BAL_COLS)

EPS = 1e-6
# Full-retrain rounds: set dynamically from Phase 1 CV mean best_iteration
# (kitchen_sink on the old 216-feature set averaged 645; the 266-feature
# set needs more -- observed ~940 on fold 0).


def combined(ll, auc):
    return 0.6 * (1 - ll) + 0.4 * auc


def evaluate(q, y):
    q = np.clip(q, EPS, 1 - EPS)
    ll = log_loss(y, q)
    auc = roc_auc_score(y, q)
    return ll, auc, combined(ll, auc)


def beta_map(p, a, b, c):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    z = a * np.log(p) - b * np.log(1 - p) + c
    return 1.0 / (1.0 + np.exp(-z))


def fit_beta(p, y):
    def nll(theta):
        a, b, c = np.exp(theta[0]), np.exp(theta[1]), theta[2]
        q = np.clip(beta_map(p, a, b, c), 1e-15, 1 - 1e-15)
        return -(y * np.log(q) + (1 - y) * np.log(1 - q)).mean()
    res = minimize(nll, x0=np.zeros(3), method="L-BFGS-B",
                   options={"maxiter": 500})
    return float(np.exp(res.x[0])), float(np.exp(res.x[1])), float(res.x[2])


def temp_map(p, T):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    z = np.log(p / (1 - p)) / T
    return 1.0 / (1.0 + np.exp(-z))


def fit_temperature(p, y):
    def nll(lt):
        T = np.exp(lt[0])
        q = np.clip(temp_map(p, T), 1e-15, 1 - 1e-15)
        return -(y * np.log(q) + (1 - y) * np.log(1 - q)).mean()
    res = minimize(nll, x0=np.zeros(1), method="L-BFGS-B",
                   options={"maxiter": 500})
    return float(np.exp(res.x[0]))


def build_features():
    tr = pd.read_csv(os.path.join(DATA, "Train.csv"))
    te = pd.read_csv(os.path.join(DATA, "Test.csv"))
    tr = add_features_base(tr)
    te = add_features_base(te)
    base_cols = [c for c in tr.columns if c not in (ID_COL, TARGET)]
    bal_p25 = float(np.percentile(tr[BAL_COLS].values.astype(float), 25))
    tr, _ = add_features_expanded(tr, bal_p25)
    te, _ = add_features_expanded(te, bal_p25)
    for c in CAT_COLS:
        tr[c] = tr[c].astype("category")
        te[c] = te[c].astype("category")
    y = tr[TARGET].values
    tr, te, _ = add_target_encoding(tr, te, y)
    state = json.load(open(os.path.join(OUT, "iter6_state.json")))
    final_feats = base_cols + state["kept_new"]
    assert len(final_feats) == 266, f"expected 266 feats, got {len(final_feats)}"
    missing = [f for f in final_feats if f not in tr.columns]
    assert not missing, f"missing features: {missing[:5]}"
    return tr, te, y, final_feats


def phase1_raw_oof(tr, te, y, feats):
    """5-fold CV, RAW oof (no per-fold calibration). Returns oof_raw, test_raw_avg."""
    X, Xte = tr[feats], te[feats]
    oof_raw = np.zeros(len(X))
    test_raw = np.zeros(len(Xte))
    iters = []
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (trn, val) in enumerate(skf.split(X, y)):
        dtr = lgb.Dataset(X.iloc[trn], y[trn], categorical_feature=CAT_COLS)
        dva = lgb.Dataset(X.iloc[val], y[val], categorical_feature=CAT_COLS,
                          reference=dtr)
        m = lgb.train(dict(KITCHEN_SINK), dtr, num_boost_round=5000,
                      valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        oof_raw[val] = m.predict(X.iloc[val], num_iteration=m.best_iteration)
        test_raw += m.predict(Xte, num_iteration=m.best_iteration) / N_SPLITS
        iters.append(m.best_iteration)
        print(f"  fold {fold}: best_iter={m.best_iteration}", flush=True)
    print(f"Phase 1 done: mean best_iter={np.mean(iters):.1f}", flush=True)
    pd.DataFrame({"ID": tr[ID_COL], "oof_pred_raw": oof_raw,
                  TARGET: y}).to_csv(os.path.join(OUT, "oof_iter7_raw.csv"),
                                     index=False)
    return oof_raw, test_raw, iters


def phase2_bakeoff(oof_raw, y):
    p = np.clip(oof_raw, 1e-15, 1 - 1e-15)
    rows = []

    ll, auc, sc = evaluate(p, y)
    rows.append(("none", ll, auc, sc, "-"))
    print(f"(a) none            LL={ll:.5f} AUC={auc:.5f} comb={sc:.5f}", flush=True)

    iso = IsotonicRegression(out_of_bounds="clip")
    q_iso = iso.fit_transform(p, y)
    ll, auc, sc = evaluate(q_iso, y)
    rows.append(("pooled_isotonic", ll, auc, sc, "isotonic on raw pooled OOF"))
    print(f"(b) pooled isotonic LL={ll:.5f} AUC={auc:.5f} comb={sc:.5f}", flush=True)

    a, b, c = fit_beta(p, y)
    ll, auc, sc = evaluate(beta_map(p, a, b, c), y)
    rows.append(("beta", ll, auc, sc, f"a={a:.4f},b={b:.4f},c={c:.4f}"))
    print(f"(c) beta calib      LL={ll:.5f} AUC={auc:.5f} comb={sc:.5f} "
          f"[a={a:.4f} b={b:.4f} c={c:.4f}]", flush=True)

    T = fit_temperature(p, y)
    ll, auc, sc = evaluate(temp_map(p, T), y)
    rows.append(("temperature", ll, auc, sc, f"T={T:.4f}"))
    print(f"(d) temperature     LL={ll:.5f} AUC={auc:.5f} comb={sc:.5f} "
          f"[T={T:.4f}]", flush=True)

    winner = max(rows, key=lambda r: r[3])
    print(f"\nWINNER: {winner[0]} comb={winner[3]:.5f}", flush=True)

    # iso.f_ may be an interp1d (newer sklearn); evaluate explicitly
    iso_y = np.asarray(iso.predict(np.asarray(iso.X_thresholds_)))
    cal = {"winner": winner[0], "beta": [a, b, c], "T": T,
           "iso_x": np.asarray(iso.X_thresholds_).tolist(),
           "iso_f": iso_y.tolist()}
    with open(os.path.join(OUT, "iter7_calibrator.json"), "w") as f:
        json.dump(cal, f)

    rep = ["# iter7 calibration bake-off (raw pooled OOF, n=40000)",
           "# NOTE: bake-off on oof_iter6.csv (already per-fold calibrated) was",
           "# degenerate -- all four calibrators converged to identity (0.81140).",
           "# This is the honest rerun on RAW pooled OOF.",
           "calibrator,logloss,auc,combined,params"]
    for name, ll_, auc_, sc_, prm in rows:
        rep.append(f"{name},{ll_:.6f},{auc_:.6f},{sc_:.6f},{prm}")
    rep.append(f"winner,{winner[0]},,,")
    with open(os.path.join(OUT, "iter7_calibration.txt"), "w") as f:
        f.write("\n".join(rep) + "\n")
    return cal


def apply_calibrator(p_raw, cal):
    p = np.clip(np.asarray(p_raw, dtype=float), 1e-15, 1 - 1e-15)
    w = cal["winner"]
    if w == "none":
        return p
    if w == "pooled_isotonic":
        # np.interp clips out-of-bounds to the end values,
        # matching IsotonicRegression(out_of_bounds="clip")
        return np.interp(p, np.array(cal["iso_x"]), np.array(cal["iso_f"]))
    if w == "beta":
        a, b, c = cal["beta"]
        return beta_map(p, a, b, c)
    if w == "temperature":
        return temp_map(p, cal["T"])
    raise ValueError(w)


def phase3_final_model(tr, te, y, feats, cal, n_rounds):
    X, Xte = tr[feats], te[feats]
    dall = lgb.Dataset(X, y, categorical_feature=CAT_COLS)
    m = lgb.train(dict(KITCHEN_SINK), dall, num_boost_round=n_rounds)
    raw = m.predict(Xte)
    q = np.clip(apply_calibrator(raw, cal), EPS, 1 - EPS)
    sub = pd.DataFrame({ID_COL: te[ID_COL], "Target": q})
    sub.to_csv(os.path.join(OUT, "iter7.csv"), index=False)
    print(f"Saved submissions/iter7.csv via '{cal['winner']}' "
          f"(test mean {q.mean():.4f}, min {q.min():.2e}, max {q.max():.4f})",
          flush=True)
    assert len(sub) == 30000 and sub["ID"].is_unique
    assert ((q > 0) & (q < 1)).all() and not np.isnan(q).any()


def main():
    print("Building features (verbatim iter6 pipeline)...", flush=True)
    tr, te, y, feats = build_features()
    print(f"features: {len(feats)} (expected 266)", flush=True)

    raw_path = os.path.join(OUT, "oof_iter7_raw.csv")
    if os.path.exists(raw_path):
        print("Phase 1: resuming from saved raw OOF...", flush=True)
        rdf = pd.read_csv(raw_path)
        oof_raw = rdf["oof_pred_raw"].values.astype(float)
        assert (rdf["ID"].values == tr[ID_COL].values).all()
        # mean best_iter from the original run's log (folds 940/584/637/770/760)
        full_rounds = 738
        print(f"Full-retrain rounds set to CV mean best_iter = {full_rounds}",
              flush=True)
    else:
        print("Phase 1: 5-fold CV, raw OOF...", flush=True)
        oof_raw, _, iters = phase1_raw_oof(tr, te, y, feats)
        full_rounds = int(round(np.mean(iters)))
        print(f"Full-retrain rounds set to CV mean best_iter = {full_rounds}",
              flush=True)

    print("Phase 2: calibration bake-off on raw pooled OOF...", flush=True)
    cal = phase2_bakeoff(oof_raw, y)

    print(f"Phase 3: full-data retrain ({full_rounds} rounds)...", flush=True)
    phase3_final_model(tr, te, y, feats, cal, full_rounds)
    print("iter7 complete. NOT submitted (daily limit reached).", flush=True)


if __name__ == "__main__":
    main()
