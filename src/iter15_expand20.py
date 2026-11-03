"""Iter15 20-seed expansion (parent directive: go to 20 seeds, not 15).

Chains after the 15-seed expansion: waits for the 15-seed stage to finish
(all 30 npz + completion marker in iter15_expand15.log), trains 5 new seeds
1001/2002/3003/4004/5005 (LGBM + CatBoost-d8, tag_prefix='iter15'; already-
trained seeds are never retrained — npz caches are reused), then runs the
hybrid blend bake-off over all 20 seeds. Overwrites submissions/iter15.csv
only if the 20-seed blend beats the current best.

Prints WINNER_20SEED / NO_IMPROVEMENT at the end for the monitor.
NEVER submits (parent handles browser). Resume-safe: seed trainers cache npz.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iter14
import iter15
from iter6 import OUT, ID_COL, EPS
from iter8 import score_oof
from iter12 import blend_bakeoff
from iter15 import BAR, SEEDS6, CB_D8, build_399, add_features_v5, load_state
from iter15 import save_state as save15

SEEDS15 = SEEDS6 + [99, 1234, 2024, 31337, 123, 456, 789, 2026, 55555]
SEEDS20 = SEEDS15 + [1001, 2002, 3003, 4004, 5005]
EXPAND_LOG = os.path.join(OUT, "iter15_expand20.log")
PRIOR_LOG = os.path.join(OUT, "iter15_expand15.log")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(EXPAND_LOG, "a") as f:
        f.write(line + "\n")


def wait_for_15seed(timeout_h=8):
    """Wait until the 15-seed expansion finished (completion marker in its
    log) and all 30 npz files exist."""
    need = ([f"iter15_lgbm_s{s}.npz" for s in SEEDS15]
            + [f"iter15_catboost_s{s}.npz" for s in SEEDS15])
    t0 = time.time()
    while time.time() - t0 < timeout_h * 3600:
        missing = [f for f in need
                   if not os.path.exists(os.path.join(OUT, f))]
        done_marker = False
        if os.path.exists(PRIOR_LOG):
            with open(PRIOR_LOG) as f:
                tail = f.read()
            done_marker = ("WINNER_15SEED" in tail) or ("NO_IMPROVEMENT" in tail)
        if not missing and done_marker:
            log("15-seed stage complete, all 30 npz present")
            return True
        if not missing and (time.time() - t0 > 3600):
            log("30 npz present but no completion marker after 1h; "
                "proceeding anyway")
            return True
        log(f"waiting for 15-seed stage: {len(missing)} npz missing, "
            f"marker={done_marker}")
        time.sleep(300)
    log("TIMEOUT waiting for 15-seed stage")
    return False


def current_best():
    st = load_state()
    w = st.get("winner")
    p = os.path.join(OUT, "iter15.csv")
    if os.path.exists(p) and w:
        return w["score"], p
    return BAR, None


def main():
    os.makedirs(OUT, exist_ok=True)
    log("expand20: waiting for 15-seed stage completion")
    if not wait_for_15seed():
        log("NO_IMPROVEMENT (15-seed stage never finished)")
        return

    st = load_state()
    kept_v5 = st.get("stage1_kept", [])
    tr399, te399, y, feats399, _ = build_399(seed=42)
    tr0, te0, _ = add_features_v5(tr399.copy(), te399.copy())
    feats_final = feats399 + kept_v5
    te_ids = te399[ID_COL].values
    log(f"feats_final={len(feats_final)} (v5 kept: {len(kept_v5)})")

    for s in [1001, 2002, 3003, 4004, 5005]:
        oof_l, _ = iter14.train_lgbm_seed(s, tr0, te0, y, feats_final,
                                          tag_prefix="iter15")
        oof_c, _ = iter14.train_cb_seed(s, tr0, te0, y, feats_final, CB_D8,
                                        tag_prefix="iter15")
        log(f"seed {s}: done LGBM={score_oof(y, oof_l):.5f} "
            f"CB={score_oof(y, oof_c):.5f}")

    oofs_l = [np.load(os.path.join(OUT, f"iter15_lgbm_s{s}.npz"))["oof"]
              for s in SEEDS20]
    tests_l = [np.load(os.path.join(OUT, f"iter15_lgbm_s{s}.npz"))["test"]
               for s in SEEDS20]
    oofs_c = [np.load(os.path.join(OUT, f"iter15_catboost_s{s}.npz"))["oof"]
              for s in SEEDS20]
    tests_c = [np.load(os.path.join(OUT, f"iter15_catboost_s{s}.npz"))["test"]
               for s in SEEDS20]
    nm, oof_b, test_b, sc_b = blend_bakeoff(y, oofs_l, tests_l, oofs_c,
                                           tests_c, SEEDS20, "iter15-20seed")
    best_sc, best_path = current_best()
    log(f"20-seed blend: hybrid:{nm} comb={sc_b:.5f} vs current best "
        f"{best_sc:.5f}")
    if sc_b > best_sc + 1e-6:
        test_pred = np.clip(test_b, 1e-6, 1 - 1e-6)
        path = os.path.join(OUT, "iter15.csv")
        pd.DataFrame({ID_COL: te_ids, "Target": test_pred}).to_csv(path,
                                                                  index=False)
        spec = {"kind": f"hybrid:{nm}", "blend": nm, "seeds": SEEDS20,
                "feats": len(feats_final), "cb_cfg": CB_D8, "n20": True}
        with open(os.path.join(OUT, "iter15_results.txt"), "w") as f:
            f.write(f"ITERATION 15 (iter15) results\nbar (iter14 10-seed): {BAR}\n")
            f.write(f"FINAL OOF combined={sc_b:.5f} BEATS {BAR}\n")
            f.write(f"spec: {json.dumps(spec)}\n")
        st["winner"] = {"score": sc_b, "file": path}
        save15(st)
        log(f"WINNER_20SEED: {path} comb={sc_b:.5f}")
    else:
        log("NO_IMPROVEMENT: 20-seed blend did not beat current best")


if __name__ == "__main__":
    main()
