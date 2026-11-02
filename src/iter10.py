"""Iteration 10: add XGBoost diversity to the iter9 3-seed ensemble.

iter9 = seed-avg of hybrid0.3geo(LGBM, CatBoost), 3 seeds, OOF 0.81472.
The blend won because LGBM and CatBoost disagree (friend's lesson).
XGBoost (one-hot categoricals, hist tree method) is a third disagreeing
learner. Pipeline per seed s (same scheme: 5-fold, per-fold isotonic):
  - LGBM kitchen_sink, CatBoost (2000/depth6): reuse iter9 npz (seeds 42,7,2025)
  - XGBoost (lr 0.03, depth 6, 2000 rounds, ES 100): train per seed
Blend bake-off over the 3 models (2-way arith grid, 3-way Nelder-Mead,
2-way/3-way geo and geo/arith hybrids), pick best OOF combined.
Bar: iter9 OOF 0.81472. Resume-safe per-seed npz cache.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iter6
import iter8
import xgboost as xgb
from iter6 import (add_features_base, add_features_expanded, add_target_encoding,
                   run_lgbm, KITCHEN_SINK, CAT_COLS, N_SPLITS, TARGET, ID_COL,
                   DATA, OUT, EPS, combined)
from iter8 import add_liquidity_features
from iter9 import hybrid03geo

BAR = 0.81472
SEEDS = [42, 7, 2025]
CAT_ORDER = ["gender", "region", "smartphone", "segment", "earning_pattern"]


def run_xgb(X, y, Xte, seed):
    """5-fold CV XGBoost (one-hot categoricals) + per-fold isotonic."""
    Xx = pd.get_dummies(X, columns=CAT_ORDER).astype(np.float32)
    Xtex = pd.get_dummies(Xte, columns=CAT_ORDER).reindex(
        columns=Xx.columns, fill_value=0).astype(np.float32)
    oof = np.zeros(len(X))
    test = np.zeros(len(Xte))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for fold, (trn, val) in enumerate(skf.split(Xx, y)):
        dtr = xgb.DMatrix(Xx.iloc[trn], label=y[trn])
        dva = xgb.DMatrix(Xx.iloc[val], label=y[val])
        params = {"objective": "binary:logistic", "eval_metric": "logloss",
                  "eta": 0.03, "max_depth": 6, "min_child_weight": 20,
                  "subsample": 0.8, "colsample_bytree": 0.8,
                  "lambda": 3.0, "alpha": 0.5, "tree_method": "hist",
                  "seed": seed, "nthread": 4}
        m = xgb.train(params, dtr, num_boost_round=2000,
                      evals=[(dva, "val")], early_stopping_rounds=100,
                      verbose_eval=False)
        pv = m.predict(dva, iteration_range=(0, m.best_iteration + 1))
        pt = m.predict(xgb.DMatrix(Xtex),
                       iteration_range=(0, m.best_iteration + 1))
        iso = IsotonicRegression(out_of_bounds="clip")
        oof[val] = np.clip(iso.fit_transform(pv, y[val]), EPS, 1 - EPS)
        test += np.clip(iso.transform(np.clip(pt, 0, 1)), EPS, 1 - EPS) / N_SPLITS
        print(f"  xgb seed {seed} fold {fold}: best_iter={m.best_iteration}",
              flush=True)
    return oof, test


def build_features():
    tr = pd.read_csv(os.path.join(DATA, "Train.csv"))
    te = pd.read_csv(os.path.join(DATA, "Test.csv"))
    tr = add_features_base(tr)
    te = add_features_base(te)
    base_cols = [c for c in tr.columns if c not in (ID_COL, TARGET)]
    bal_p25 = float(np.percentile(
        tr[[f"m{i}_daily_avg_bal" for i in range(6, 0, -1)]].values.astype(float), 25))
    tr, _ = add_features_expanded(tr, bal_p25)
    te, _ = add_features_expanded(te, bal_p25)
    tr, liq_new = add_liquidity_features(tr)
    te, _ = add_liquidity_features(te)
    for c in CAT_COLS:
        tr[c] = tr[c].astype("category")
        te[c] = te[c].astype("category")
    with open(os.path.join(OUT, "iter6_state.json")) as f:
        kept_new = json.load(f)["kept_new"]
    feats = base_cols + kept_new + liq_new
    missing = [c for c in feats if c not in tr.columns]
    assert not missing, f"missing: {missing}"
    print(f"features: {len(feats)}", flush=True)
    return tr, te, tr[TARGET].values, feats


def score(y, p):
    return combined(log_loss(y, p), roc_auc_score(y, p))


def main():
    os.makedirs(OUT, exist_ok=True)
    tr0, te0, y, feats = build_features()

    oofs = {"lgbm": [], "catboost": [], "xgb": []}
    tests = {"lgbm": [], "catboost": [], "xgb": []}
    # reuse iter9 caches for lgbm/catboost
    for s in SEEDS:
        d = np.load(os.path.join(OUT, f"iter9_lgbm_s{s}.npz")) if s != 42 else \
            np.load(os.path.join(OUT, "iter8_lgbm.npz"))
        oofs["lgbm"].append(d["oof"]); tests["lgbm"].append(d["test"])
        d = np.load(os.path.join(OUT, f"iter9_catboost_s{s}.npz")) if s != 42 else \
            np.load(os.path.join(OUT, "iter8_catboost.npz"))
        oofs["catboost"].append(d["oof"]); tests["catboost"].append(d["test"])

    for s in SEEDS:
        iter6.SEED = s
        p = os.path.join(OUT, f"iter10_xgb_s{s}.npz")
        if os.path.exists(p):
            d = np.load(p)
            oof_x, test_x = d["oof"], d["test"]
            print(f"seed {s}: XGB loaded cached", flush=True)
        else:
            tr, te = tr0.copy(), te0.copy()
            tr, te, _ = add_target_encoding(tr, te, y)
            print(f"seed {s}: training XGBoost...", flush=True)
            oof_x, test_x = run_xgb(tr[feats], y, te[feats], s)
            np.savez(p, oof=oof_x, test=test_x)
            print(f"seed {s}: XGB done, comb={score(y, oof_x):.5f}", flush=True)
        oofs["xgb"].append(oof_x); tests["xgb"].append(test_x)

    # seed-average per model
    avg = {m: np.mean(oofs[m], axis=0) for m in oofs}
    avg_t = {m: np.mean(tests[m], axis=0) for m in tests}
    for m in oofs:
        print(f"{m:8s} seed-avg comb={score(y, np.clip(avg[m], EPS, 1-EPS)):.5f}",
              flush=True)

    # ---- blend bake-off (lgbm, catboost, xgb) ----
    L, C, X = avg["lgbm"], avg["catboost"], avg["xgb"]
    Lt, Ct, Xt = avg_t["lgbm"], avg_t["catboost"], avg_t["xgb"]
    cands = {}  # name -> (oof, test)

    def reg(nm, po, pt):
        cands[nm] = (po, pt)

    models = [(L, Lt), (C, Ct), (X, Xt)]
    names = ["lgbm", "catboost", "xgb"]
    # arithmetic 2-way grid
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        for w in [0.3, 0.5, 0.7]:
            po = w * models[i][0] + (1 - w) * models[j][0]
            pt = w * models[i][1] + (1 - w) * models[j][1]
            reg(f"arith_{names[i]}{w:.1f}_{names[j]}{1-w:.1f}", po, pt)
    # 3-way Nelder-Mead via softmax
    def neg_comb(z):
        w = np.exp(z); w = w / w.sum()
        p = sum(wi * m[0] for wi, m in zip(w, models))
        return -score(y, np.clip(p, EPS, 1 - EPS))
    res = minimize(neg_comb, np.zeros(3), method="Nelder-Mead",
                   options={"maxiter": 300, "xatol": 1e-3, "fatol": 1e-6})
    w = np.exp(res.x); w = w / w.sum()
    nm_oof = sum(wi * m[0] for wi, m in zip(w, models))
    nm_test = sum(wi * m[1] for wi, m in zip(w, models))
    reg(f"nm3w_{'_'.join(f'{x:.2f}' for x in w)}", nm_oof, nm_test)
    # geometric / hybrid 2-way (lgbm x catboost, lgbm x xgb, catboost x xgb)
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        a, b = models[i][0], models[j][0]
        at, bt = models[i][1], models[j][1]
        g = np.sqrt(np.clip(a, EPS, 1) * np.clip(b, EPS, 1))
        gt = np.sqrt(np.clip(at, EPS, 1) * np.clip(bt, EPS, 1))
        reg(f"geo_{names[i]}_{names[j]}", g, gt)
        for hw in [0.3, 0.5]:
            po = hw * g + (1 - hw) * (0.5 * a + 0.5 * b)
            pt = hw * gt + (1 - hw) * (0.5 * at + 0.5 * bt)
            reg(f"hybrid{hw:.1f}geo_{names[i]}_{names[j]}", po, pt)
    # 3-way geometric
    g3 = (np.clip(L, EPS, 1) * np.clip(C, EPS, 1) * np.clip(X, EPS, 1)) ** (1/3)
    g3t = (np.clip(Lt, EPS, 1) * np.clip(Ct, EPS, 1) * np.clip(Xt, EPS, 1)) ** (1/3)
    reg("geo3", g3, g3t)
    # iter9 recipe as candidate
    hyb_oof, hyb_test = hybrid03geo(L, C), hybrid03geo(Lt, Ct)
    reg("iter9_hybrid_lgbm_cb", hyb_oof, hyb_test)

    scored = []
    for nm, (po, pt) in cands.items():
        po = np.clip(po, EPS, 1 - EPS)
        sc = score(y, po)
        scored.append((nm, sc, log_loss(y, po), roc_auc_score(y, po)))
    scored.sort(key=lambda r: -r[1])
    print("\nBlend candidates (OOF combined):", flush=True)
    for nm, sc, ll, auc in scored:
        print(f"  {nm:34s} LL={ll:.5f} AUC={auc:.5f} comb={sc:.5f}", flush=True)

    best_nm, best_sc = scored[0][0], scored[0][1]
    best_oof, best_test = cands[best_nm]
    print(f"\nFINAL: {best_nm} OOF comb={best_sc:.5f} "
          f"({'BEATS' if best_sc > BAR else 'below'} iter9 bar {BAR})", flush=True)

    with open(os.path.join(OUT, "iter10_results.txt"), "w") as f:
        f.write(f"seeds: {SEEDS}\n")
        for nm, sc, ll, auc in scored:
            f.write(f"  {nm:34s} LL={ll:.5f} AUC={auc:.5f} comb={sc:.5f}\n")
        f.write(f"FINAL: {best_nm} OOF comb={best_sc:.5f} (iter9 bar {BAR})\n")

    if best_sc > BAR:
        best_test = np.clip(best_test, EPS, 1 - EPS)
        pd.DataFrame({ID_COL: te0[ID_COL], "Target": best_test}).to_csv(
            os.path.join(OUT, "iter10.csv"), index=False)
        pd.DataFrame({"ID": tr0[ID_COL], "oof_pred": np.clip(best_oof, EPS, 1-EPS),
                      TARGET: y}).to_csv(os.path.join(OUT, "oof_iter10.csv"),
                                         index=False)
        print(f"Saved submissions/iter10.csv (test mean {best_test.mean():.4f}). "
              f"BEATS bar — ready to submit.", flush=True)
    else:
        print("Below bar: iter10.csv NOT written.", flush=True)
    print("EXIT:0")


if __name__ == "__main__":
    main()
