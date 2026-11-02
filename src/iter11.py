"""Iteration 11: extend iter9's 3-seed ensemble to 6 seeds.

iter9 = seed-avg of hybrid0.3geo(LGBM, CatBoost) over seeds 42/7/2025,
OOF 0.81472. Seed averaging keeps helping (lgbm seed-avg 0.81353 >
single-seed 0.81080-0.81108). Add seeds 5, 11, 777 for LGBM + CatBoost,
reuse all cached seeds, and re-pick between recipe A (avg of seed hybrids)
and B (hybrid of seed-averaged models). Bar: 0.81472.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iter6
import iter8
from iter6 import (add_features_base, add_features_expanded, add_target_encoding,
                   run_lgbm, KITCHEN_SINK, CAT_COLS, TARGET, ID_COL,
                   DATA, OUT, EPS)
from iter8 import add_liquidity_features, score_oof
from iter9 import hybrid03geo, build_features, train_seed

BAR = 0.81472
SEED42, OLD = 42, [7, 2025]
NEW = [5, 11, 777]


def main():
    os.makedirs(OUT, exist_ok=True)
    tr0, te0, y, feats = build_features()

    oofs_l, tests_l, oofs_c, tests_c = [], [], [], []
    d = np.load(os.path.join(OUT, "iter8_lgbm.npz"))
    oofs_l.append(d["oof"]); tests_l.append(d["test"])
    d = np.load(os.path.join(OUT, "iter8_catboost.npz"))
    oofs_c.append(d["oof"]); tests_c.append(d["test"])

    for s in OLD + NEW:
        oof_l, test_l, oof_c, test_c = train_seed(s, tr0, te0, y, feats)
        oofs_l.append(oof_l); tests_l.append(test_l)
        oofs_c.append(oof_c); tests_c.append(test_c)

    seeds = [SEED42] + OLD + NEW
    hyb_oofs = [hybrid03geo(oofs_l[i], oofs_c[i]) for i in range(len(seeds))]
    hyb_tests = [hybrid03geo(tests_l[i], tests_c[i]) for i in range(len(seeds))]
    for s, h in zip(seeds, hyb_oofs):
        print(f"seed {s}: hybrid OOF comb={score_oof(y, h):.5f}", flush=True)

    oof_A = np.mean(hyb_oofs, axis=0)
    test_A = np.mean(hyb_tests, axis=0)
    sc_A = score_oof(y, np.clip(oof_A, EPS, 1 - EPS))
    oof_B = hybrid03geo(np.mean(oofs_l, axis=0), np.mean(oofs_c, axis=0))
    test_B = hybrid03geo(np.mean(tests_l, axis=0), np.mean(tests_c, axis=0))
    sc_B = score_oof(y, np.clip(oof_B, EPS, 1 - EPS))
    print(f"candidate A (avg of seed hybrids): {sc_A:.5f}", flush=True)
    print(f"candidate B (hybrid of avg models): {sc_B:.5f}", flush=True)

    if sc_A >= sc_B:
        oof_f, test_f, pick = oof_A, test_A, "seed-avg-of-hybrids-6seed"
    else:
        oof_f, test_f, pick = oof_B, test_B, "hybrid-of-seed-avgs-6seed"
    final = max(sc_A, sc_B)
    print(f"FINAL: {pick} OOF comb={final:.5f} "
          f"({'BEATS' if final > BAR else 'below'} bar {BAR})", flush=True)

    with open(os.path.join(OUT, "iter11_results.txt"), "w") as f:
        f.write(f"seeds: {seeds}\n")
        f.write(f"A: {sc_A:.5f}\nB: {sc_B:.5f}\nFINAL: {pick} {final:.5f} "
                f"(bar {BAR})\n")

    if final > BAR:
        test_f = np.clip(test_f, EPS, 1 - EPS)
        pd.DataFrame({ID_COL: te0[ID_COL], "Target": test_f}).to_csv(
            os.path.join(OUT, "iter11.csv"), index=False)
        print(f"Saved submissions/iter11.csv (test mean {test_f.mean():.4f}). "
              f"BEATS bar — ready to submit.", flush=True)
    else:
        print("Below bar: iter11.csv NOT written.", flush=True)
    print("EXIT:0")


if __name__ == "__main__":
    main()
