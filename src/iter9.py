"""Iteration 9: multi-seed ensemble of the iter8 winner.

iter8 = hybrid0.3geo(CatBoost, LightGBM) on 279 features, OOF combined 0.81283
(seed 42, 5-fold, per-fold isotonic). Variance across seeds is free gain:
average the seed-42 blend with the same pipeline trained at seeds 7 and 2025.

Pipeline per seed s:
- feature set identical to iter8 (iter6 266 feats + 13 liquidity interactions);
- target encoding folds, LGBM folds, CatBoost folds/splits all seeded by s;
- per-fold isotonic calibration (same scheme as iter6/iter8);
- seed OOF = hybrid0.3geo(lgbm_s, catboost_s) [iter8's winning recipe].

Final candidates (OOF combined on the shared 5-fold structure of each seed):
  A: mean over seeds of per-seed hybrid OOF
  B: hybrid0.3geo(mean lgbm OOF, mean catboost OOF)
Pick the better; write submissions/iter9.csv only if it beats the iter8 bar
0.81283. Resume-safe: per-seed per-model npz files are skipped when present.
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iter6
import iter8
from iter6 import (add_features_base, add_features_expanded, add_target_encoding,
                   run_lgbm, KITCHEN_SINK, CAT_COLS, N_SPLITS, TARGET, ID_COL,
                   DATA, OUT, EPS)
from iter8 import add_liquidity_features, score_oof

BAR = 0.81283  # iter8 OOF combined
SEED42 = 42
EXTRA_SEEDS = [7, 2025]


def hybrid03geo(a, b):
    g = np.sqrt(np.clip(a, EPS, 1) * np.clip(b, EPS, 1))
    return 0.3 * g + 0.7 * (0.5 * a + 0.5 * b)


def build_features():
    """Heavy feature engineering done once; target encoding is per-seed."""
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
    assert not missing, f"missing features: {missing}"
    y = tr[TARGET].values
    print(f"features: {len(feats)} (base {len(base_cols)} + kept {len(kept_new)} "
          f"+ liq {len(liq_new)})", flush=True)
    return tr, te, y, feats


def train_seed(seed, tr0, te0, y, feats):
    """Train LGBM + CatBoost at `seed`; return (oof_lgbm, test_lgbm, oof_cb, test_cb)."""
    iter6.SEED = seed
    iter8.SEED = seed
    tr = tr0.copy()
    te = te0.copy()
    tr, te, _ = add_target_encoding(tr, te, y)
    X, Xte = tr[feats], te[feats]

    p_lgbm = os.path.join(OUT, f"iter9_lgbm_s{seed}.npz")
    if os.path.exists(p_lgbm):
        d = np.load(p_lgbm)
        oof_l, test_l = d["oof"], d["test"]
        print(f"seed {seed}: LGBM loaded cached", flush=True)
    else:
        print(f"seed {seed}: training LGBM...", flush=True)
        params = dict(KITCHEN_SINK, seed=seed)
        oof_l, test_l, mi = run_lgbm(X, y, Xte, params)
        np.savez(p_lgbm, oof=oof_l, test=test_l, iters=mi)
        print(f"seed {seed}: LGBM done, mean_iter={mi:.0f}, "
              f"comb={score_oof(y, oof_l):.5f}", flush=True)

    p_cb = os.path.join(OUT, f"iter9_catboost_s{seed}.npz")
    if os.path.exists(p_cb):
        d = np.load(p_cb)
        oof_c, test_c = d["oof"], d["test"]
        print(f"seed {seed}: CatBoost loaded cached", flush=True)
    else:
        print(f"seed {seed}: training CatBoost (2000 iters)...", flush=True)
        oof_c, test_c, mi = iter8.run_catboost(X, y, Xte)
        np.savez(p_cb, oof=oof_c, test=test_c, iters=mi)
        print(f"seed {seed}: CatBoost done, mean_iter={mi:.0f}, "
              f"comb={score_oof(y, oof_c):.5f}", flush=True)
    return oof_l, test_l, oof_c, test_c


def main():
    os.makedirs(OUT, exist_ok=True)
    tr0, te0, y, feats = build_features()

    oofs_l, tests_l, oofs_c, tests_c = [], [], [], []
    # seed 42: reuse iter8 cache (same pipeline, same seed)
    for name, m in (("lgbm", "lgbm"), ("catboost", "catboost")):
        d = np.load(os.path.join(OUT, f"iter8_{name}.npz"))
        if m == "lgbm":
            oofs_l.append(d["oof"]); tests_l.append(d["test"])
        else:
            oofs_c.append(d["oof"]); tests_c.append(d["test"])
    print(f"seed {SEED42}: LGBM comb={score_oof(y, oofs_l[0]):.5f}, "
          f"CatBoost comb={score_oof(y, oofs_c[0]):.5f} (iter8 cache)", flush=True)

    for s in EXTRA_SEEDS:
        oof_l, test_l, oof_c, test_c = train_seed(s, tr0, te0, y, feats)
        oofs_l.append(oof_l); tests_l.append(test_l)
        oofs_c.append(oof_c); tests_c.append(test_c)

    seeds = [SEED42] + EXTRA_SEEDS
    # per-seed hybrid OOFs
    hyb_oofs, hyb_tests = [], []
    for i, s in enumerate(seeds):
        h_oof = hybrid03geo(oofs_l[i], oofs_c[i])
        h_te = hybrid03geo(tests_l[i], tests_c[i])
        hyb_oofs.append(h_oof); hyb_tests.append(h_te)
        print(f"seed {s}: hybrid OOF comb={score_oof(y, h_oof):.5f}", flush=True)

    # candidate A: average of per-seed hybrids
    oof_A = np.mean(hyb_oofs, axis=0)
    test_A = np.mean(hyb_tests, axis=0)
    sc_A = score_oof(y, np.clip(oof_A, EPS, 1 - EPS))
    # candidate B: hybrid of seed-averaged models
    oof_B = hybrid03geo(np.mean(oofs_l, axis=0), np.mean(oofs_c, axis=0))
    test_B = hybrid03geo(np.mean(tests_l, axis=0), np.mean(tests_c, axis=0))
    sc_B = score_oof(y, np.clip(oof_B, EPS, 1 - EPS))
    print(f"candidate A (avg of seed hybrids): {sc_A:.5f}", flush=True)
    print(f"candidate B (hybrid of avg models): {sc_B:.5f}", flush=True)

    if sc_A >= sc_B:
        oof_final, test_final, pick = oof_A, test_A, "seed-avg-of-hybrids"
    else:
        oof_final, test_final, pick = oof_B, test_B, "hybrid-of-seed-avgs"
    final_score = max(sc_A, sc_B)
    print(f"FINAL: {pick} OOF comb={final_score:.5f} "
          f"({'BEATS' if final_score > BAR else 'below'} iter8 bar {BAR})", flush=True)

    with open(os.path.join(OUT, "iter9_results.txt"), "w") as f:
        f.write(f"seeds: {seeds}\n")
        f.write(f"candidate A (avg of seed hybrids): {sc_A:.5f}\n")
        f.write(f"candidate B (hybrid of avg models): {sc_B:.5f}\n")
        f.write(f"FINAL: {pick} OOF comb={final_score:.5f} (iter8 bar {BAR})\n")

    if final_score > BAR:
        oof_final = np.clip(oof_final, EPS, 1 - EPS)
        test_final = np.clip(test_final, EPS, 1 - EPS)
        pd.DataFrame({ID_COL: te0[ID_COL], "Target": test_final}).to_csv(
            os.path.join(OUT, "iter9.csv"), index=False)
        pd.DataFrame({"ID": tr0[ID_COL], "oof_pred": oof_final,
                      TARGET: y}).to_csv(os.path.join(OUT, "oof_iter9.csv"),
                                         index=False)
        print(f"Saved submissions/iter9.csv (test mean {test_final.mean():.4f}). "
              f"BEATS bar — ready to submit.", flush=True)
    else:
        print("Below bar: iter9.csv NOT written.", flush=True)
    print("EXIT:0")


if __name__ == "__main__":
    main()
