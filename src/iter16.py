"""Iteration 16: v6 battery with HONEST (customer-grouped) validation.

CRITICAL FIX: train = 10k customers x 4 snapshots; test = 10k x 3. Our
row-wise stratified 5-fold put the same customer in train AND validation,
inflating CV (0.82543) and misranking ideas (v5's 0/79 was judged by the
biased validator). This script rebuilds everything on StratifiedGroupKFold
by customer.

Step 0: honest OOF harness; re-score iter14's 399-feature baseline (grouped
        LGBM + grouped CatBoost, seed 42) -> honest bar.
Step 1: greedy forward selection of 116 v6 candidates on the honest bar
        (same batch protocol as v5, but grouped CV).
Step 2: if v6 wins -> 10-seed grouped hybrid blend -> submissions/iter16.csv.

Protocol: 5-fold grouped CV, per-fold isotonic, combined = 0.6*(1-LL)+0.4*AUC.
Resume-safe via submissions/iter16_state.json + per-seed npz. NEVER submits
(parent handles browser). Commit, don't push.
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.isotonic import IsotonicRegression
from catboost import CatBoostClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iter6
import iter8
import iter12
import iter14
import iter15
from iter6 import (KITCHEN_SINK, CAT_COLS, N_SPLITS, TARGET, ID_COL, DATA,
                   OUT, EPS, combined)
from iter8 import score_oof, CAT_ORDER
from iter12 import blend_bakeoff
from v6_battery import add_features_v6

PROF_COLS = ["arpu", "age", "gender", "region", "smartphone", "segment",
             "earning_pattern", "x_90_d_activity_rate"]
CB_D8 = {"depth": 8, "learning_rate": 0.03, "iterations": 2000,
         "l2_leaf_reg": 3.0}   # won iter14's retest
SEEDS10 = [42, 7, 2025, 5, 11, 777, 99, 1234, 2024, 31337]
BATCH = 20
PATIENCE = 2
MAX_FEATURES = 460
STATE16 = os.path.join(OUT, "iter16_state.json")


def save_state(st):
    with open(STATE16, "w") as f:
        json.dump(st, f)


def load_state():
    if os.path.exists(STATE16):
        with open(STATE16) as f:
            return json.load(f)
    return {}


def customer_groups(df):
    """10,000 customer identities from raw profile columns (4 rows each)."""
    return pd.factorize(df[PROF_COLS].apply(tuple, axis=1))[0]


def sgkf(seed=42):
    return StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                random_state=seed)


def add_te_grouped(tr, te, y, groups, seed=42):
    """5-fold OOF smoothed target encoding with GROUPED folds (no customer
    leakage); full-train smoothed means for test. Overwrites te_* columns."""
    gm = float(y.mean())
    for c in CAT_COLS:
        oof = np.full(len(tr), gm)
        for trn, val in sgkf(seed).split(tr, y, groups):
            ktr = tr[c].iloc[trn].astype(str).values
            kval = tr[c].iloc[val].astype(str).values
            yt = y[trn]
            df_tmp = pd.DataFrame({"k": ktr, "y": yt})
            agg = df_tmp.groupby("k")["y"].agg(["mean", "size"])
            smooth = (agg["size"] * agg["mean"] + 10 * gm) / (agg["size"] + 10)
            oof[val] = pd.Series(kval).map(smooth).fillna(gm).values
        tr[f"te_{c}"] = oof
        ktr_all = tr[c].astype(str).values
        df_tmp = pd.DataFrame({"k": ktr_all, "y": y})
        agg = df_tmp.groupby("k")["y"].agg(["mean", "size"])
        smooth = (agg["size"] * agg["mean"] + 10 * gm) / (agg["size"] + 10)
        te[f"te_{c}"] = te[c].astype(str).map(smooth).fillna(gm).values
    return tr, te


def run_lgbm_grouped(X, y, Xte, params, groups, seed=42):
    """Grouped 5-fold CV LGBM + per-fold isotonic. Returns oof, test, mean_iter."""
    oof = np.zeros(len(X))
    test = np.zeros(len(Xte))
    iters = []
    for trn, val in sgkf(seed).split(X, y, groups):
        dtr = lgb.Dataset(X.iloc[trn], y[trn], categorical_feature=CAT_COLS)
        dva = lgb.Dataset(X.iloc[val], y[val], categorical_feature=CAT_COLS,
                          reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        pv = m.predict(X.iloc[val], num_iteration=m.best_iteration)
        pt = m.predict(Xte, num_iteration=m.best_iteration)
        iso = IsotonicRegression(out_of_bounds="clip")
        oof[val] = np.clip(iso.fit_transform(pv, y[val]), 1e-6, 1 - 1e-6)
        test += np.clip(iso.transform(np.clip(pt, 0, 1)), 1e-6, 1 - 1e-6) / N_SPLITS
        iters.append(m.best_iteration)
    return oof, test, float(np.mean(iters))


def run_cb_grouped(X, y, Xte, cfg, seed, groups):
    """Grouped 5-fold CV CatBoost(d8) + per-fold isotonic. Returns oof, test."""
    oof = np.zeros(len(X))
    test = np.zeros(len(Xte))
    iters = []
    for fold, (trn, val) in enumerate(sgkf(seed).split(X, y, groups)):
        Xtr = X.iloc[trn].copy()
        Xva = X.iloc[val].copy()
        Xte_c = Xte.copy()
        for c in CAT_ORDER:
            Xtr[c] = Xtr[c].astype(str)
            Xva[c] = Xva[c].astype(str)
            Xte_c[c] = Xte[c].astype(str)
        m = CatBoostClassifier(iterations=cfg["iterations"],
                               learning_rate=cfg["learning_rate"],
                               depth=cfg["depth"],
                               l2_leaf_reg=cfg["l2_leaf_reg"],
                               loss_function="Logloss", eval_metric="Logloss",
                               random_seed=seed, early_stopping_rounds=100,
                               use_best_model=True, verbose=False,
                               allow_writing_files=False, thread_count=4)
        m.fit(Xtr, y[trn], eval_set=(Xva, y[val]), cat_features=CAT_ORDER)
        pv = m.predict_proba(Xva)[:, 1]
        pt = m.predict_proba(Xte_c)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        oof[val] = np.clip(iso.fit_transform(pv, y[val]), EPS, 1 - EPS)
        test += np.clip(iso.transform(np.clip(pt, 0, 1)), EPS, 1 - EPS) / N_SPLITS
        iters.append(m.get_best_iteration())
    return oof, test, float(np.mean(iters))


def build_399_honest(seed=42):
    """iter14's 399-feature set, but with GROUPED target encoding (overwrites
    the leaky row-wise TE from build_359). Returns tr, te, y, feats, groups."""
    tr399, te399, y, feats399, bal_p25 = iter15.build_399(seed=seed)
    groups = customer_groups(pd.read_csv(os.path.join(DATA, "Train.csv")))
    assert len(np.unique(groups)) == 10000, "group count != 10000"
    tr399, te399 = add_te_grouped(tr399, te399, y, groups, seed=seed)
    print(f"399-feat honest set (seed {seed}): {len(feats399)} feats, "
          f"{len(np.unique(groups))} customer groups", flush=True)
    return tr399, te399, y, feats399, groups


def add_v6(tr, te):
    """Attach the 116 v6 candidates. Returns (tr, te, v6_names)."""
    dtr, dte = add_features_v6(tr, te)
    names = list(dtr.keys())
    for k in names:
        tr[k] = dtr[k]
        te[k] = dte[k]
    assert all(np.isfinite(tr[n]).all() and np.isfinite(te[n]).all()
               for n in names)
    print(f"v6 battery attached: {len(names)} candidates", flush=True)
    return tr, te, names


# ---------------------------------------------------------------- Step 0
def step0_honest_baseline(tr, te, y, feats399, groups):
    """Grouped seed-42 LGBM + CatBoost on 399 feats -> honest bar."""
    print("\n=== Step 0: honest baseline (grouped CV, 399 feats) ===", flush=True)
    st = load_state()
    if st.get("step0_done"):
        print(f"step0 cached: lgbm={st['step0_lgbm']:.5f} "
              f"cb={st['step0_cb']:.5f} hybrid={st['step0_hybrid']:.5f}",
              flush=True)
        return st["step0_lgbm"], st["step0_cb"], st["step0_hybrid"]

    oof_l, test_l, mi_l = run_lgbm_grouped(
        tr[feats399], y, te[feats399], dict(KITCHEN_SINK, seed=42), groups, 42)
    sc_l = score_oof(y, oof_l)
    print(f"  honest LGBM(399, seed42): comb={sc_l:.5f} (leaky was 0.82163)",
          flush=True)
    oof_c, test_c, mi_c = run_cb_grouped(
        tr[feats399], y, te[feats399], CB_D8, 42, groups)
    sc_c = score_oof(y, oof_c)
    print(f"  honest CatBoost(399, seed42): comb={sc_c:.5f}", flush=True)
    hyb = np.clip(0.5 * np.sqrt(np.clip(oof_l, EPS, 1) * np.clip(oof_c, EPS, 1))
                  + 0.5 * (0.5 * oof_l + 0.5 * oof_c), EPS, 1 - EPS)
    sc_h = score_oof(y, hyb)
    print(f"  honest hybrid(399, seed42): comb={sc_h:.5f} "
          f"(leaky 10-seed was 0.82543)", flush=True)
    st.update({"step0_done": True, "step0_lgbm": sc_l, "step0_cb": sc_c,
               "step0_hybrid": sc_h})
    save_state(st)
    return sc_l, sc_c, sc_h


# ---------------------------------------------------------------- Step 1
def step1_greedy_v6(tr, te, y, feats399, v6_names, groups, bar):
    """Greedy forward selection of v6 candidates in gain-ranked batches of 20,
    judged by grouped-CV LGBM vs the honest bar."""
    print("\n=== Step 1: greedy v6 screen on HONEST bar ===", flush=True)
    st = load_state()
    if st.get("step1_done"):
        print(f"step1 cached: kept {len(st['step1_kept'])} best {st['step1_best']:.5f}",
              flush=True)
        return st["step1_kept"], st["step1_best"]

    if "ranked_v6" not in st:
        print("gain-ranking fit on full data (399+v6)...", flush=True)
        Xg = tr[feats399 + v6_names]
        dall = lgb.Dataset(Xg, y, categorical_feature=CAT_COLS)
        m = lgb.train(dict(KITCHEN_SINK, num_threads=4), dall,
                      num_boost_round=800)
        gain = pd.Series(m.feature_importance(importance_type="gain"),
                         index=feats399 + v6_names)
        st["ranked_v6"] = [f for f in gain.sort_values(ascending=False).index
                           if f in v6_names]
        save_state(st)
        print("  top-10 v6 by gain:", st["ranked_v6"][:10], flush=True)
    ranked_v6 = st["ranked_v6"]

    kept = st.get("step1_kept", [])
    done_batches = set(st.get("step1_batches", []))
    best = st.get("step1_best", bar)
    print(f"step1 honest bar (grouped LGBM 399): {bar:.5f}", flush=True)

    batches = [ranked_v6[i:i + BATCH] for i in range(0, len(ranked_v6), BATCH)]
    no_gain = st.get("step1_nogain", 0)
    for bi, batch in enumerate(batches):
        if bi in done_batches:
            print(f"batch {bi}: skipped (done)", flush=True)
            continue
        if len(feats399) + len(kept) + len(batch) > MAX_FEATURES:
            print(f"batch {bi}: skipped (would exceed {MAX_FEATURES})", flush=True)
            done_batches.add(bi)
            continue
        cand = feats399 + kept + batch
        oof, _, _ = run_lgbm_grouped(tr[cand], y, te[cand],
                                     dict(KITCHEN_SINK, seed=42), groups, 42)
        sc = score_oof(y, oof)
        improved = sc > best + 1e-6
        if improved:
            kept.extend(batch)
            best = sc
            no_gain = 0
        else:
            no_gain += 1
        print(f"batch {bi} (+{len(batch)} -> {len(cand)} total): comb={sc:.5f} "
              f"{'KEPT' if improved else 'dropped'} (best {best:.5f})", flush=True)
        done_batches.add(bi)
        st.update({"step1_kept": kept, "step1_batches": sorted(done_batches),
                   "step1_best": best, "step1_nogain": no_gain})
        save_state(st)
        if no_gain >= PATIENCE:
            print(f"early stop: {PATIENCE} non-improving batches", flush=True)
            break
    st["step1_done"] = True
    save_state(st)
    print(f"Step 1 done: kept {len(kept)} v6 features, best={best:.5f}", flush=True)
    return kept, best


# ---------------------------------------------------------------- Step 2
def train_lgbm_seed_grouped(seed, tr0, te0, y, feats, groups, tag="iter16"):
    p = os.path.join(OUT, f"{tag}_lgbm_s{seed}.npz")
    if os.path.exists(p):
        d = np.load(p)
        print(f"seed {seed}: LGBM loaded cached", flush=True)
        return d["oof"], d["test"]
    tr = tr0.copy()
    te = te0.copy()
    tr, te = add_te_grouped(tr, te, y, groups, seed=seed)
    prm = dict(KITCHEN_SINK, seed=seed)
    print(f"seed {seed}: training grouped LGBM...", flush=True)
    oof, test, mi = run_lgbm_grouped(tr[feats], y, te[feats], prm, groups, seed)
    np.savez(p, oof=oof, test=test, iters=mi)
    print(f"seed {seed}: LGBM done, mean_iter={mi:.0f}, comb={score_oof(y, oof):.5f}",
          flush=True)
    return oof, test


def train_cb_seed_grouped(seed, tr0, te0, y, feats, groups, tag="iter16"):
    p = os.path.join(OUT, f"{tag}_catboost_s{seed}.npz")
    if os.path.exists(p):
        d = np.load(p)
        print(f"seed {seed}: CatBoost loaded cached", flush=True)
        return d["oof"], d["test"]
    tr = tr0.copy()
    te = te0.copy()
    tr, te = add_te_grouped(tr, te, y, groups, seed=seed)
    print(f"seed {seed}: training grouped CatBoost {CB_D8}...", flush=True)
    oof, test, mi = run_cb_grouped(tr[feats], y, te[feats], CB_D8, seed, groups)
    np.savez(p, oof=oof, test=test, iters=mi)
    print(f"seed {seed}: CatBoost done, mean_iter={mi:.0f}, comb={score_oof(y, oof):.5f}",
          flush=True)
    return oof, test


def step2_final_blend(tr0, te0, y, feats_final, groups, te_ids, honest_bar):
    """10-seed grouped hybrid blend; saves iter16.csv if it beats honest bar."""
    print(f"\n=== Step 2: 10-seed grouped assembly + blend "
          f"({len(feats_final)} feats) ===", flush=True)
    oofs_l, tests_l, oofs_c, tests_c = [], [], [], []
    for s in SEEDS10:
        oof_l, test_l = train_lgbm_seed_grouped(s, tr0, te0, y, feats_final,
                                                groups)
        oof_c, test_c = train_cb_seed_grouped(s, tr0, te0, y, feats_final,
                                              groups)
        oofs_l.append(oof_l)
        tests_l.append(test_l)
        oofs_c.append(oof_c)
        tests_c.append(test_c)
    nm, oof_b, test_b, sc_b = blend_bakeoff(y, oofs_l, tests_l, oofs_c,
                                            tests_c, SEEDS10, "iter16-final")
    print(f"\nStep 2 final: hybrid:{nm} honest OOF comb={sc_b:.5f} "
          f"({'BEATS' if sc_b > honest_bar else 'below'} honest bar {honest_bar:.5f})",
          flush=True)
    path = None
    if sc_b > honest_bar:
        test_pred = np.clip(test_b, 1e-6, 1 - 1e-6)
        path = os.path.join(OUT, "iter16.csv")
        pd.DataFrame({ID_COL: te_ids, "Target": test_pred}).to_csv(path,
                                                                   index=False)
        print(f"Saved submissions/iter16.csv (test mean {test_pred.mean():.4f}, "
              f"honest OOF comb {sc_b:.5f})", flush=True)
        with open(os.path.join(OUT, "iter16_results.txt"), "w") as f:
            f.write("ITERATION 16 results (HONEST customer-grouped CV)\n")
            f.write(f"honest bar: {honest_bar:.5f}\n")
            f.write(f"FINAL honest OOF combined={sc_b:.5f} BEATS {honest_bar:.5f}\n")
            f.write(f"spec: hybrid:{nm} seeds={SEEDS10} feats={len(feats_final)}\n")
    st = load_state()
    st["step2"] = {"score": sc_b, "path": path, "spec": nm}
    save_state(st)
    return sc_b, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="0,1,2",
                    help="comma-separated subset of stages to run")
    args = ap.parse_args()
    stages = set(args.stages.split(","))
    os.makedirs(OUT, exist_ok=True)
    st = load_state()

    tr399, te399, y, feats399, groups = build_399_honest(seed=42)
    te_ids = te399[ID_COL].values

    honest_bar = None
    if "0" in stages:
        sc_l, sc_c, sc_h = step0_honest_baseline(tr399, te399, y, feats399,
                                                groups)
        honest_bar = sc_l  # screening bar = grouped seed-42 LGBM
        st["honest_bar_lgbm"] = sc_l
        st["honest_bar_hybrid"] = sc_h
        save_state(st)
    else:
        honest_bar = st.get("honest_bar_lgbm")
        assert honest_bar, "run stage 0 first (or state missing)"

    kept_v6, best1 = [], honest_bar
    if "1" in stages:
        tr6, te6, v6_names = add_v6(tr399.copy(), te399.copy())
        kept_v6, best1 = step1_greedy_v6(tr6, te6, y, feats399, v6_names,
                                        groups, honest_bar)
    elif st.get("step1_done"):
        kept_v6 = st["step1_kept"]
        best1 = st["step1_best"]
    feats_final = feats399 + kept_v6
    print(f"\nfeats_final={len(feats_final)} (v6 kept: {len(kept_v6)})",
          flush=True)

    # full feature frame for step 2 (v6 attached even if 0 kept: harmless)
    tr0, te0, _ = add_v6(tr399.copy(), te399.copy())

    winner_path, winner_score = None, 0.0
    if "2" in stages:
        if best1 > honest_bar + 1e-6:
            sc2, path2 = step2_final_blend(tr0, te0, y, feats_final, groups,
                                           te_ids, honest_bar)
            if path2:
                winner_path, winner_score = path2, sc2
        else:
            print(f"\nNo v6 gain on honest bar ({best1:.5f} vs {honest_bar:.5f}); "
                  f"skipping 10-seed assembly.", flush=True)

    if winner_path:
        print(f"\nWINNER: {winner_path} honest comb={winner_score:.5f} > "
              f"bar {honest_bar:.5f}", flush=True)
    else:
        print(f"\nNo winner this run (honest bar {honest_bar:.5f}).", flush=True)
    print("EXIT:0")


if __name__ == "__main__":
    main()
