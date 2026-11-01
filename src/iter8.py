"""Iteration 8: friend's winning strategy — CatBoost + liquidity interactions + blend.

Context: a friend's pipeline hit public 0.71882 with (1) CatBoost (2000 iters,
lr 0.03, depth 6, early stopping 100, native categoricals) on v2-features +
liquidity-pressure interactions (OOF AUC 0.91030 / LL 0.24479), then
(2) 0.5*CatBoost + 0.5*v4-ensemble blend (OOF AUC 0.91227 / LL 0.24287).
The blend worked because the models disagreed.

Our iter5 CatBoost flopped (0.80340) — weak features + undertuned. This time
it gets our full iter6 266-feature set PLUS new liquidity interactions.

Pipeline (comparable scheme throughout):
- 5-fold stratified CV, seed 42, per-fold isotonic calibration,
  combined = 0.6*(1-LogLoss) + 0.4*ROC-AUC.
- (a) LightGBM kitchen_sink on 266 + ~11 new liquidity features.
- (b) CatBoost (iterations=2000, lr=0.03, depth=6, early_stopping=100,
      Logloss, random_seed=42), 5 categoricals passed NATIVELY.
- (c) HistGradientBoostingClassifier (lr=0.05, max_iter=1000,
      early_stopping=True), categoricals one-hot encoded.
- Blend bake-off on calibrated OOF: 0.5/0.5, weight grid, Nelder-Mead
  (2-way & 3-way), geometric-mean blends, geo/arith hybrid.
- Final: blend member test preds (fold-averaged, calibrated) with winning
  weights; fallback to best single model if no blend helps.

RESUME-SAFE: per-model OOF/test saved to submissions/iter8_<model>.npz;
completed models are skipped on restart. Blend + final are cheap.

Bar: iter6 CV combined 0.81140 on the same seed-42 scheme.
Output: submissions/iter8.csv, oof_iter8.csv (NOT submitted here —
daily limit reached; file left ready for tomorrow's reset).
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iter6 import (add_features_base, add_features_expanded, add_target_encoding,
                   run_lgbm, KITCHEN_SINK, CAT_COLS, combined,
                   SEED, N_SPLITS, TARGET, ID_COL, DATA, OUT)

ITER6_BAR = 0.81140
EPS = 1e-6
CAT_ORDER = ["gender", "region", "smartphone", "segment", "earning_pattern"]


def add_liquidity_features(df):
    """Liquidity-pressure interactions (friend's strategy, adapted).

    Skips exact duplicates of existing features (netflow_m1, netflow_sum).
    Returns (df, new_col_names).
    """
    df = df.copy()
    new = {}

    def monthly(metric):
        return df[[f"m{i}_{metric}" for i in range(1, 7)]].values.astype(float)

    bal = monthly("daily_avg_bal")            # (n,6) m1..m6 newest-first
    w_tv = monthly("withdraw_total_value")
    d_tv = monthly("deposit_total_value")
    r_tv = monthly("received_total_value")
    s_tv = monthly("mm_send_total_value")
    pb_tv = monthly("paybill_total_value")
    mp_tv = monthly("merchantpay_total_value")

    # 1. latest-month withdrawals relative to balance
    new["liq_withdraw_pressure_m1"] = w_tv[:, 0] / np.maximum(bal[:, 0], 1.0)

    # 2. total value (all categories, all months) relative to mean balance
    all_tv = w_tv + d_tv + r_tv + s_tv + pb_tv + mp_tv + monthly("transfer_from_bank_total_value")
    new["liq_total_value_div_balmean"] = all_tv.sum(axis=1) / np.maximum(np.abs(bal.mean(axis=1)), 1.0)

    # 3. netflow m1/total with the friend's definition == base netflow_m1/netflow_sum
    #    -> exact duplicates, skipped (kept for documentation, not added).

    # 4. balance drop from max x withdrawal share (6m)
    withdraw_share_6m = df["withdraw_share_6m"].values
    new["liq_bal_drop_x_withdraw_share"] = (
        (df["bal_max"].values - bal[:, 0]) * withdraw_share_6m)

    # 5. balance minus latest-month withdrawals
    new["liq_bal_minus_withdraw_m1"] = bal[:, 0] - w_tv[:, 0]

    # 6. static interactions
    new["liq_arpu_x_activity"] = df["arpu"].values * df["x_90_d_activity_rate"].values
    new["liq_age_x_arpu"] = df["age"].values * df["arpu"].values

    # 7. withdrawal pressure ratio vs deposits, latest month
    new["liq_withdraw_pressure_ratio_m1"] = w_tv[:, 0] / np.maximum(d_tv[:, 0], 1.0)

    # 8. months-of-runway per month (balance / that month's outflow)
    outflow = w_tv + s_tv + pb_tv + mp_tv
    for i in range(6):
        new[f"liq_runway_m{i + 1}"] = bal[:, i] / np.maximum(outflow[:, i], 1.0)

    for k, v in new.items():
        if k in df.columns:
            raise RuntimeError(f"name collision: {k}")
        df[k] = v
    return df, list(new.keys())


def run_catboost(X, y, Xte):
    """5-fold CV CatBoost with native categoricals + per-fold isotonic."""
    oof = np.zeros(len(X))
    test = np.zeros(len(Xte))
    iters = []
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (trn, val) in enumerate(skf.split(X, y)):
        Xtr = X.iloc[trn].copy()
        Xva = X.iloc[val].copy()
        Xte_c = Xte.copy()
        for c in CAT_ORDER:
            Xtr[c] = Xtr[c].astype(str)
            Xva[c] = Xva[c].astype(str)
            Xte_c[c] = Xte_c[c].astype(str)
        m = CatBoostClassifier(iterations=2000, learning_rate=0.03, depth=6,
                               loss_function="Logloss", eval_metric="Logloss",
                               random_seed=SEED, early_stopping_rounds=100,
                               use_best_model=True, verbose=False,
                               allow_writing_files=False)
        m.fit(Xtr, y[trn], eval_set=(Xva, y[val]), cat_features=CAT_ORDER)
        pv = m.predict_proba(Xva)[:, 1]
        pt = m.predict_proba(Xte_c)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        oof[val] = np.clip(iso.fit_transform(pv, y[val]), EPS, 1 - EPS)
        test += np.clip(iso.transform(np.clip(pt, 0, 1)), EPS, 1 - EPS) / N_SPLITS
        iters.append(m.get_best_iteration())
        print(f"  catboost fold {fold}: best_iter={iters[-1]}", flush=True)
    return oof, test, float(np.mean(iters))


def run_histgb(X, y, Xte):
    """5-fold CV sklearn HistGradientBoosting (one-hot categoricals)."""
    Xh = pd.get_dummies(X, columns=CAT_ORDER)
    Xteh = pd.get_dummies(Xte, columns=CAT_ORDER).reindex(columns=Xh.columns, fill_value=0)
    oof = np.zeros(len(X))
    test = np.zeros(len(Xte))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (trn, val) in enumerate(skf.split(X, y)):
        m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=1000,
                                           early_stopping=True, validation_fraction=0.1,
                                           n_iter_no_change=50, random_state=SEED)
        m.fit(Xh.iloc[trn].values, y[trn])
        pv = m.predict_proba(Xh.iloc[val].values)[:, 1]
        pt = m.predict_proba(Xteh.values)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        oof[val] = np.clip(iso.fit_transform(pv, y[val]), EPS, 1 - EPS)
        test += np.clip(iso.transform(np.clip(pt, 0, 1)), EPS, 1 - EPS) / N_SPLITS
        print(f"  histgb fold {fold}: n_iter={m.n_iter_}", flush=True)
    return oof, test


def score_oof(y, p):
    return combined(log_loss(y, p), roc_auc_score(y, p))


def blend_candidates(oofs, names):
    """Return dict name -> (weights or spec, blended_oof, score)."""
    y_global = None
    cands = {}

    def reg(nm, p):
        cands[nm] = p

    # arithmetic grids
    if len(oofs) >= 2:
        for w in [0.3, 0.5, 0.7]:
            reg(f"arith_cb{w:.1f}_lgbm{1 - w:.1f}",
                w * oofs[1] + (1 - w) * oofs[0])
    if len(oofs) >= 3:
        # Nelder-Mead 3-way via softmax
        def neg_comb(z):
            w = np.exp(z)
            w = w / w.sum()
            p = sum(wi * o for wi, o in zip(w, oofs))
            return -score_oof(Y, p)
        res = minimize(neg_comb, np.zeros(len(oofs)), method="Nelder-Mead",
                       options={"maxiter": 200, "xatol": 1e-3, "fatol": 1e-6})
        w = np.exp(res.x)
        w = w / w.sum()
        reg(f"nm3w_{'_'.join(f'{x:.2f}' for x in w)}",
            sum(wi * o for wi, o in zip(w, oofs)))
    # geometric blends
    if len(oofs) >= 2:
        g2 = np.sqrt(np.clip(oofs[0], EPS, 1) * np.clip(oofs[1], EPS, 1))
        reg("geo_cb_lgbm", g2)
        for w in [0.3, 0.5, 0.7]:  # hybrid w*geo + (1-w)*arith(0.5/0.5)
            a = 0.5 * oofs[0] + 0.5 * oofs[1]
            reg(f"hybrid{w:.1f}geo_cb_lgbm", w * g2 + (1 - w) * a)
    if len(oofs) >= 3:
        g3 = (np.clip(oofs[0], EPS, 1) * np.clip(oofs[1], EPS, 1)
              * np.clip(oofs[2], EPS, 1)) ** (1.0 / 3.0)
        reg("geo3", g3)
    return cands


Y = None  # set in main for blend tuning closure


def main():
    global Y
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
    print(f"liquidity features added: {len(liq_new)}", flush=True)

    for c in CAT_COLS:
        tr[c] = tr[c].astype("category")
        te[c] = te[c].astype("category")
    y = tr[TARGET].values
    Y = y
    tr, te, new_te = add_target_encoding(tr, te, y)

    # iter6 final feature set = base_cols + kept_new (from state)
    with open(os.path.join(OUT, "iter6_state.json")) as f:
        kept_new = json.load(f)["kept_new"]
    feats = base_cols + kept_new + liq_new
    # sanity: all must exist
    missing = [c for c in feats if c not in tr.columns]
    assert not missing, f"missing features: {missing}"
    print(f"features: base {len(base_cols)} + kept {len(kept_new)} + liq {len(liq_new)} "
          f"= {len(feats)}", flush=True)
    X, Xte = tr[feats], te[feats]
    print(f"X shape {X.shape}, Xte shape {Xte.shape}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    results = {}

    # ---- (a) LightGBM kitchen_sink ----
    p_lgbm = os.path.join(OUT, "iter8_lgbm.npz")
    if os.path.exists(p_lgbm):
        d = np.load(p_lgbm)
        oof_lgbm, test_lgbm = d["oof"], d["test"]
        print("LGBM: loaded cached", flush=True)
    else:
        print("LGBM kitchen_sink training...", flush=True)
        oof_lgbm, test_lgbm, mi = run_lgbm(X, y, Xte, KITCHEN_SINK)
        np.savez(p_lgbm, oof=oof_lgbm, test=test_lgbm, iters=mi)
    results["lgbm"] = score_oof(y, oof_lgbm)
    print(f"LGBM combined={results['lgbm']:.5f} "
          f"(LL={log_loss(y, oof_lgbm):.5f} AUC={roc_auc_score(y, oof_lgbm):.5f})", flush=True)

    # ---- (b) CatBoost ----
    p_cb = os.path.join(OUT, "iter8_catboost.npz")
    if os.path.exists(p_cb):
        d = np.load(p_cb)
        oof_cb, test_cb = d["oof"], d["test"]
        print("CatBoost: loaded cached", flush=True)
    else:
        print("CatBoost training (2000 iters, depth 6)...", flush=True)
        oof_cb, test_cb, mi = run_catboost(X, y, Xte)
        np.savez(p_cb, oof=oof_cb, test=test_cb, iters=mi)
    results["catboost"] = score_oof(y, oof_cb)
    print(f"CatBoost combined={results['catboost']:.5f} "
          f"(LL={log_loss(y, oof_cb):.5f} AUC={roc_auc_score(y, oof_cb):.5f})", flush=True)

    # ---- (c) HistGB ----
    p_hg = os.path.join(OUT, "iter8_histgb.npz")
    if os.path.exists(p_hg):
        d = np.load(p_hg)
        oof_hg, test_hg = d["oof"], d["test"]
        print("HistGB: loaded cached", flush=True)
    else:
        print("HistGB training...", flush=True)
        oof_hg, test_hg = run_histgb(X, y, Xte)
        np.savez(p_hg, oof=oof_hg, test=test_hg)
    results["histgb"] = score_oof(y, oof_hg)
    print(f"HistGB combined={results['histgb']:.5f} "
          f"(LL={log_loss(y, oof_hg):.5f} AUC={roc_auc_score(y, oof_hg):.5f})", flush=True)

    # ---- blend bake-off (order: lgbm, catboost, histgb) ----
    oofs = [oof_lgbm, oof_cb, oof_hg]
    tests = [test_lgbm, test_cb, test_hg]
    names = ["lgbm", "catboost", "histgb"]
    cands = blend_candidates(oofs, names)
    scored = []
    for nm, p in cands.items():
        p = np.clip(p, EPS, 1 - EPS)
        sc = score_oof(y, p)
        scored.append((nm, sc, log_loss(y, p), roc_auc_score(y, p)))
    scored.sort(key=lambda r: -r[1])
    print("\nBlend candidates (OOF combined):", flush=True)
    for nm, sc, ll, auc in scored:
        print(f"  {nm:28s} LL={ll:.5f} AUC={auc:.5f} comb={sc:.5f}", flush=True)

    best_single = max(results.items(), key=lambda kv: kv[1])
    best_blend = scored[0]
    print(f"\nBest single: {best_single[0]} {best_single[1]:.5f} | "
          f"Best blend: {best_blend[0]} {best_blend[1]:.5f} | "
          f"iter6 bar: {ITER6_BAR}", flush=True)

    # ---- final pick ----
    if best_blend[1] > best_single[1] + 1e-6:
        # reconstruct winning blend on test preds
        win_nm = best_blend[0]
        win_oof = cands[win_nm]
        # recompute test blend with the same recipe
        if win_nm.startswith("arith_cb"):
            w = float(win_nm.split("_")[1][2:])
            test_final = w * test_cb + (1 - w) * test_lgbm
            spec = {"type": "arith", "w_cb": w, "w_lgbm": 1 - w}
        elif win_nm.startswith("nm3w_"):
            w = np.array([float(x) for x in win_nm.split("_")[1:]])
            test_final = sum(wi * t for wi, t in zip(w, tests))
            spec = {"type": "nm3w", "weights": w.tolist(), "models": names}
        elif win_nm == "geo_cb_lgbm":
            test_final = np.sqrt(np.clip(test_lgbm, EPS, 1) * np.clip(test_cb, EPS, 1))
            spec = {"type": "geo2"}
        elif win_nm.startswith("hybrid"):
            w = float(win_nm.split("hybrid")[1].split("geo")[0])
            g = np.sqrt(np.clip(test_lgbm, EPS, 1) * np.clip(test_cb, EPS, 1))
            a = 0.5 * test_lgbm + 0.5 * test_cb
            test_final = w * g + (1 - w) * a
            spec = {"type": "hybrid", "w_geo": w}
        elif win_nm == "geo3":
            test_final = (np.clip(test_lgbm, EPS, 1) * np.clip(test_cb, EPS, 1)
                          * np.clip(test_hg, EPS, 1)) ** (1.0 / 3.0)
            spec = {"type": "geo3"}
        else:
            raise RuntimeError(f"unknown blend {win_nm}")
        oof_final = np.clip(win_oof, EPS, 1 - EPS)
        pick = f"blend:{win_nm}"
    else:
        idx = names.index(best_single[0])
        test_final = tests[idx]
        oof_final = oofs[idx]
        spec = {"type": "single", "model": best_single[0]}
        pick = f"single:{best_single[0]}"
    final_score = score_oof(y, oof_final)
    print(f"FINAL PICK: {pick} OOF comb={final_score:.5f} "
          f"{'BEATS' if final_score > ITER6_BAR else 'below'} iter6 bar {ITER6_BAR}",
          flush=True)

    test_final = np.clip(test_final, EPS, 1 - EPS)
    pd.DataFrame({ID_COL: te[ID_COL], "Target": test_final}).to_csv(
        os.path.join(OUT, "iter8.csv"), index=False)
    pd.DataFrame({"ID": tr[ID_COL], "oof_pred": oof_final, TARGET: y}).to_csv(
        os.path.join(OUT, "oof_iter8.csv"), index=False)
    with open(os.path.join(OUT, "iter8_blend.json"), "w") as f:
        json.dump({"pick": pick, "spec": spec,
                   "oof_combined": final_score,
                   "per_model": results,
                   "blends": [{"name": nm, "combined": sc, "ll": ll, "auc": auc}
                              for nm, sc, ll, auc in scored]}, f, indent=2)
    with open(os.path.join(OUT, "iter8_results.txt"), "w") as f:
        f.write(f"per-model OOF combined: {results}\n")
        f.write("blend candidates:\n")
        for nm, sc, ll, auc in scored:
            f.write(f"  {nm:28s} LL={ll:.5f} AUC={auc:.5f} comb={sc:.5f}\n")
        f.write(f"FINAL PICK: {pick} OOF comb={final_score:.5f} "
                f"(iter6 bar {ITER6_BAR})\n")
    print(f"Saved submissions/iter8.csv (test mean {test_final.mean():.4f}). "
          f"NOT submitted (daily limit reached).", flush=True)

    # top new liquidity features by LGBM gain (quick single full-data fit)
    d_all = lgb.Dataset(X, y, categorical_feature=CAT_COLS)
    m = lgb.train(dict(KITCHEN_SINK, num_threads=4), d_all, num_boost_round=800)
    gain = pd.Series(m.feature_importance(importance_type="gain"), index=feats)
    top_liq = [(f, g) for f, g in gain.sort_values(ascending=False).items()
               if f in liq_new][:10]
    print("Top liquidity features by gain:", flush=True)
    for f, g in top_liq:
        print(f"  {f:38s} {g:,.0f}", flush=True)
    with open(os.path.join(OUT, "iter8_blend.json"), "r+") as f:
        j = json.load(f)
        j["top_liq_features"] = [f for f, _ in top_liq]
        f.seek(0)
        json.dump(j, f, indent=2)
        f.truncate()


if __name__ == "__main__":
    main()
