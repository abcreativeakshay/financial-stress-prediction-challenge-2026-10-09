"""Iteration 19: HVC attack (research direction #3).

HVC (high-value customers, segment == 'HVC') is the worst residual pocket:
3,736 train rows (9.3%), LL 0.3218 / AUC 0.8753 vs MVC 0.2048 / 0.9246 on
the iter18 base OOF. Full fix worth ~+0.0066 OOF; realistic partial
+0.001-0.003. HVC is a RANKING deficit, so recalibration cannot fix it --
only a modeling change can.

Two members, both on the SAME honest customer-grouped validation
(StratifiedGroupKFold on customer; never row-wise CV), same 439-feature
set as iter16/iter18 (399 base + 40 kept v6), same per-fold isotonic
protocol, combined = 0.6*(1-LL) + 0.4*AUC:

  A. HVC-upweighted LGBM: HVC rows get sample weight 2.5 in training
     (5 seeds: 42, 7, 2025, 5, 11). Full 40k-row OOF + test.
  B. HVC specialist LGBM: trained on HVC rows only (3,736 rows, top-120
     gain-ranked features, num_leaves=15), grouped CV within HVC
     customers. OOF/test defined on HVC rows only; blended into the base
     for HVC rows only.

Base = iter18 (0.9 * iter16 10-seed hybrid + 0.1 * RealMLP 2-seed),
reconstructed from cached npz files; iter16 seeds and RealMLP seeds are
NOT retrained. Blend sweeps: A weight w in [0.05..0.30]; specialist
weight beta on HVC rows in [0.10..0.50]; then combined best.

Gate: final blend must beat iter18's honest OOF comb 0.8256752862522517.
If it beats the bar -> submissions/iter19.csv. If not -> no file.

Resume-safe via per-seed npz (iter19_up_lgbm_s{seed}.npz,
iter19_spec_s{seed}.npz) + iter19_state.json. NEVER submits (parent
handles browser). Commit, don't push.
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iter16
from iter16 import SEEDS10, add_te_grouped, add_v6, build_399_honest, sgkf
from iter6 import (KITCHEN_SINK, CAT_COLS, N_SPLITS, TARGET, ID_COL, DATA,
                   OUT, EPS, combined)
from iter8 import score_oof, CAT_ORDER
from iter12 import hybrid
from iter17 import reconstruct_iter16_oof
from sklearn.metrics import log_loss, roc_auc_score

BAR18 = 0.8256752862522517           # iter18 honest OOF comb (w=0.10 RealMLP)
W_HVC = 2.5                          # upweight for HVC rows in member A
MEMBER_SEEDS = [42, 7, 2025, 5]  # seed 11 dropped 2026-11-06 ~02:12 UTC:
# its up-LGBM fold stalled 69+ min under CPU contention with the sibling
# VTUAD vessel job; the 4-seed up-LGBM blend was already evaluated offline
# (best w=0.05 -> 0.82567, below bar) so a 5th seed cannot change the gate.
W_GRID_A = [0.05, 0.10, 0.15, 0.20, 0.30]
BETA_GRID = [0.10, 0.20, 0.30, 0.40, 0.50]
SPEC_TOPK = 120
STATE19 = os.path.join(OUT, "iter19_state.json")


def save_state(st):
    with open(STATE19, "w") as f:
        json.dump(st, f)


def load_state():
    if os.path.exists(STATE19):
        with open(STATE19) as f:
            return json.load(f)
    return {}


def hvc_masks(tr, te):
    """HVC boolean masks for train/test (segment column is raw profile)."""
    raw_tr = pd.read_csv(os.path.join(DATA, "Train.csv"), usecols=["segment"])
    raw_te = pd.read_csv(os.path.join(DATA, "Test.csv"), usecols=["segment"])
    assert len(raw_tr) == len(tr) and len(raw_te) == len(te)
    return (raw_tr["segment"] == "HVC").values, (raw_te["segment"] == "HVC").values


def run_lgbm_upweighted(X, y, Xte, params, groups, seed, w_hvc, hvc_mask):
    """Grouped 5-fold CV LGBM + per-fold isotonic, HVC rows upweighted
    w_hvc in TRAINING folds only (val scoring stays unweighted/honest)."""
    oof = np.zeros(len(X))
    test = np.zeros(len(Xte))
    iters = []
    w = np.where(hvc_mask, w_hvc, 1.0)
    for trn, val in sgkf(seed).split(X, y, groups):
        dtr = lgb.Dataset(X.iloc[trn], y[trn], weight=w[trn],
                          categorical_feature=CAT_COLS)
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


def train_up_lgbm_seed(seed, tr0, te0, y, feats, groups, hvc_mask):
    p = os.path.join(OUT, f"iter19_up_lgbm_s{seed}.npz")
    if os.path.exists(p):
        d = np.load(p)
        print(f"seed {seed}: upweighted LGBM loaded cached", flush=True)
        return d["oof"], d["test"]
    tr = tr0.copy()
    te = te0.copy()
    tr, te = add_te_grouped(tr, te, y, groups, seed=seed)
    print(f"seed {seed}: training HVC-upweighted LGBM (w_hvc={W_HVC})...",
          flush=True)
    oof, test, mi = run_lgbm_upweighted(
        tr[feats], y, te[feats], dict(KITCHEN_SINK, seed=seed), groups, seed,
        W_HVC, hvc_mask)
    np.savez(p, oof=oof, test=test, iters=mi)
    print(f"seed {seed}: up-LGBM done, mean_iter={mi:.0f}, "
          f"comb={score_oof(y, oof):.5f}", flush=True)
    return oof, test


def spec_features(tr_hvc, y_hvc, feats):
    """Top-SPEC_TOPK features by gain from one LGBM fit on HVC rows only."""
    dall = lgb.Dataset(tr_hvc[feats], y_hvc, categorical_feature=CAT_COLS)
    m = lgb.train(dict(KITCHEN_SINK, seed=42, num_threads=4), dall,
                  num_boost_round=800)
    gain = pd.Series(m.feature_importance(importance_type="gain"), index=feats)
    top = list(gain.sort_values(ascending=False).index[:SPEC_TOPK])
    print(f"specialist top-{SPEC_TOPK} features (gain): {top[:10]}...",
          flush=True)
    return top


def run_spec_grouped(X, y, Xte, params, groups, seed):
    """Grouped 5-fold CV LGBM on HVC rows only + per-fold isotonic.
    NOTE: X holds only the top-120 specialist features, so only the
    categorical columns actually present may be declared."""
    oof = np.zeros(len(X))
    test = np.zeros(len(Xte))
    iters = []
    cat_here = [c for c in CAT_COLS if c in X.columns]
    for trn, val in StratifiedGroupKFold(
            n_splits=N_SPLITS, shuffle=True,
            random_state=seed).split(X, y, groups):
        dtr = lgb.Dataset(X.iloc[trn], y[trn], categorical_feature=cat_here)
        dva = lgb.Dataset(X.iloc[val], y[val], categorical_feature=cat_here,
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


def train_spec_seed(seed, tr0, te0, y, feats, groups, hvc_tr, hvc_te,
                    spec_feats):
    p = os.path.join(OUT, f"iter19_spec_s{seed}.npz")
    if os.path.exists(p):
        d = np.load(p)
        print(f"seed {seed}: specialist loaded cached", flush=True)
        return d["oof"], d["test"]
    tr = tr0.copy()
    te = te0.copy()
    tr, te = add_te_grouped(tr, te, y, groups, seed=seed)
    Xh = tr.loc[hvc_tr, spec_feats]
    yh = y[hvc_tr]
    gh = groups[hvc_tr]
    Xte_h = te.loc[hvc_te, spec_feats]
    print(f"seed {seed}: training HVC specialist LGBM "
          f"({len(spec_feats)} feats, {hvc_tr.sum()} rows)...", flush=True)
    oof_h, test_h, mi = run_spec_grouped(
        Xh, yh, Xte_h, dict(KITCHEN_SINK, seed=seed, num_leaves=15), gh, seed)
    np.savez(p, oof=oof_h, test=test_h, iters=mi)
    print(f"seed {seed}: specialist done, mean_iter={mi:.0f}, "
          f"HVC comb={score_oof(yh, oof_h):.5f}", flush=True)
    return oof_h, test_h


def reconstruct_base18(y):
    """iter18 base OOF + test: 0.9*iter16 hybrid + 0.1*RealMLP (2 seeds)."""
    oof16, _ = reconstruct_iter16_oof(y)
    rmlp_oof = np.clip(np.mean(
        [np.load(os.path.join(OUT, f"iter18_realmlp_s{s}.npz"))["oof"]
         for s in (42, 7)], axis=0), EPS, 1 - EPS)
    oof18 = np.clip(0.9 * oof16 + 0.1 * rmlp_oof, EPS, 1 - EPS)
    sc = score_oof(y, oof18)
    print(f"reconstructed iter18 base OOF: comb={sc:.5f} "
          f"(expected {BAR18:.5f})", flush=True)
    assert abs(sc - BAR18) < 1e-4, f"iter18 reconstruction mismatch: {sc}"
    test18 = pd.read_csv(os.path.join(OUT, "iter18.csv"))["Target"].values
    print(f"iter18 test (from csv): mean={test18.mean():.4f} "
          f"n={len(test18)}", flush=True)
    return oof18, test18, oof16, rmlp_oof


def slice_report(y, p, mask, name):
    ll = log_loss(y[mask], p[mask])
    auc = roc_auc_score(y[mask], p[mask])
    print(f"  {name}: n={mask.sum()} LL={ll:.4f} AUC={auc:.4f}", flush=True)
    return ll, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="up,spec,blend",
                    help="comma-separated subset: up,spec,blend")
    args = ap.parse_args()
    stages = set(args.stages.split(","))
    os.makedirs(OUT, exist_ok=True)
    st = load_state()

    st16 = json.load(open(os.path.join(OUT, "iter16_state.json")))
    kept_v6 = st16["step1_kept"]
    assert len(kept_v6) == 40, f"kept_v6 != 40: {len(kept_v6)}"

    tr399, te399, y, feats399, groups = build_399_honest(seed=42)
    te_ids = te399[ID_COL].values
    feats_final = feats399 + kept_v6
    print(f"\nfeats_final={len(feats_final)} (v6 kept: {len(kept_v6)})",
          flush=True)
    tr0, te0, _ = add_v6(tr399.copy(), te399.copy())
    hvc_tr, hvc_te = hvc_masks(tr399, te399)
    print(f"HVC rows: train {hvc_tr.sum()} / {len(y)}, "
          f"test {hvc_te.sum()} / {len(te399)}", flush=True)

    oof_a = test_a = None
    if "up" in stages:
        oofs, tests = [], []
        for s in MEMBER_SEEDS:
            oof_s, test_s = train_up_lgbm_seed(s, tr0, te0, y, feats_final,
                                               groups, hvc_tr)
            oofs.append(oof_s)
            tests.append(test_s)
            st["up_seeds_done"] = sorted(st.get("up_seeds_done", []) + [s])
            save_state(st)
        oof_a = np.clip(np.mean(oofs, axis=0), EPS, 1 - EPS)
        test_a = np.clip(np.mean(tests, axis=0), EPS, 1 - EPS)
        print(f"\nUpweighted LGBM ({len(MEMBER_SEEDS)}-seed mean) OOF "
              f"comb={score_oof(y, oof_a):.5f}", flush=True)
        st["up_oof_comb"] = score_oof(y, oof_a)
        save_state(st)
    else:
        oofs = [np.load(os.path.join(OUT, f"iter19_up_lgbm_s{s}.npz"))
                for s in MEMBER_SEEDS]
        oof_a = np.clip(np.mean([d["oof"] for d in oofs], axis=0), EPS, 1 - EPS)
        test_a = np.clip(np.mean([d["test"] for d in oofs], axis=0), EPS, 1 - EPS)

    oof_b = test_b = None
    if "spec" in stages:
        # gain ranking on HVC rows (seed-42 TE frame, deterministic)
        tr42, te42 = add_te_grouped(tr0.copy(), te0.copy(), y, groups, seed=42)
        spec_feats = spec_features(tr42.loc[hvc_tr], y[hvc_tr], feats_final)
        st["spec_feats"] = spec_feats
        save_state(st)
        oofs, tests = [], []
        for s in MEMBER_SEEDS:
            oof_s, test_s = train_spec_seed(s, tr0, te0, y, feats_final,
                                            groups, hvc_tr, hvc_te, spec_feats)
            oofs.append(oof_s)
            tests.append(test_s)
            st["spec_seeds_done"] = sorted(
                st.get("spec_seeds_done", []) + [s])
            save_state(st)
        oof_b = np.clip(np.mean(oofs, axis=0), EPS, 1 - EPS)
        test_b = np.clip(np.mean(tests, axis=0), EPS, 1 - EPS)
        print(f"\nSpecialist ({len(MEMBER_SEEDS)}-seed mean) HVC OOF "
              f"comb={score_oof(y[hvc_tr], oof_b):.5f}", flush=True)
        st["spec_oof_comb_hvc"] = score_oof(y[hvc_tr], oof_b)
        save_state(st)
    else:
        oofs = [np.load(os.path.join(OUT, f"iter19_spec_s{s}.npz"))
                for s in MEMBER_SEEDS]
        oof_b = np.clip(np.mean([d["oof"] for d in oofs], axis=0), EPS, 1 - EPS)
        test_b = np.clip(np.mean([d["test"] for d in oofs], axis=0), EPS, 1 - EPS)

    if "blend" in stages:
        oof18, test18, oof16, rmlp_oof = reconstruct_base18(y)
        print("\n--- per-slice report (base18 vs members) ---", flush=True)
        for nm, p in [("base18", oof18), ("up-LGBM", oof_a)]:
            print(f"{nm} overall comb={score_oof(y, p):.5f}", flush=True)
            slice_report(y, p, hvc_tr, "HVC")
            slice_report(y, p, ~hvc_tr, "non-HVC")
        print(f"specialist HVC comb={score_oof(y[hvc_tr], oof_b):.5f} "
              f"vs base18 HVC:", flush=True)
        slice_report(y, oof18, hvc_tr, "base18 HVC")
        corr_a = float(np.corrcoef(oof18, oof_a)[0, 1])
        print(f"corr(up-LGBM OOF, base18 OOF)={corr_a:.4f}", flush=True)
        st["corr_up_base18"] = corr_a

        print("\n--- blend sweep A: (1-w)*base18 + w*up-LGBM ---", flush=True)
        best_a = (0.0, score_oof(y, oof18))
        sweep_a = {}
        for w in W_GRID_A:
            oof_f = np.clip((1 - w) * oof18 + w * oof_a, EPS, 1 - EPS)
            sc = score_oof(y, oof_f)
            sweep_a[str(w)] = sc
            print(f"  w={w:.2f} comb={sc:.5f} "
                  f"({'BEATS' if sc > BAR18 else 'below'} bar {BAR18:.5f})",
                  flush=True)
            if sc > best_a[1]:
                best_a = (w, sc)
        st["sweep_a"] = sweep_a

        print("\n--- blend sweep B: HVC rows only, "
              "(1-b)*base18 + b*specialist ---", flush=True)
        best_b = (0.0, score_oof(y, oof18))
        sweep_b = {}
        for beta in BETA_GRID:
            oof_f = oof18.copy()
            oof_f[hvc_tr] = np.clip((1 - beta) * oof18[hvc_tr]
                                    + beta * oof_b, EPS, 1 - EPS)
            sc = score_oof(y, oof_f)
            sweep_b[str(beta)] = sc
            print(f"  beta={beta:.2f} comb={sc:.5f} "
                  f"({'BEATS' if sc > BAR18 else 'below'} bar {BAR18:.5f})",
                  flush=True)
            if sc > best_b[1]:
                best_b = (beta, sc)
        st["sweep_b"] = sweep_b

        # combined: best A weight + best specialist beta (HVC rows only)
        w_a, _ = best_a
        beta_b, _ = best_b
        oof_f = np.clip((1 - w_a) * oof18 + w_a * oof_a, EPS, 1 - EPS)
        if beta_b > 0:
            oof_f[hvc_tr] = np.clip(
                (1 - beta_b) * oof_f[hvc_tr] + beta_b * oof_b, EPS, 1 - EPS)
        sc_f = score_oof(y, oof_f)
        print(f"\nCombined best: w_A={w_a:.2f} beta_B={beta_b:.2f} "
              f"comb={sc_f:.5f} vs bar {BAR18:.5f}", flush=True)
        slice_report(y, oof_f, hvc_tr, "final HVC")
        st["best"] = {"w_a": w_a, "beta_b": beta_b, "comb": sc_f}
        save_state(st)

        if sc_f > BAR18:
            test_f = np.clip((1 - w_a) * test18 + w_a * test_a, 1e-6, 1 - 1e-6)
            if beta_b > 0:
                test_f[hvc_te] = np.clip(
                    (1 - beta_b) * test_f[hvc_te] + beta_b * test_b,
                    1e-6, 1 - 1e-6)
            path = os.path.join(OUT, "iter19.csv")
            pd.DataFrame({ID_COL: te_ids, "Target": test_f}).to_csv(
                path, index=False)
            print(f"Saved submissions/iter19.csv (test mean "
                  f"{test_f.mean():.4f}, honest OOF comb {sc_f:.5f})",
                  flush=True)
            with open(os.path.join(OUT, "iter19_results.txt"), "w") as f:
                f.write("ITERATION 19 results (HONEST customer-grouped CV)\n")
                f.write(f"iter18 bar: {BAR18:.5f}\n")
                f.write(f"up-LGBM (w_hvc={W_HVC}, {len(MEMBER_SEEDS)} seeds): "
                        f"OOF comb={score_oof(y, oof_a):.5f}\n")
                f.write(f"specialist HVC OOF comb="
                        f"{score_oof(y[hvc_tr], oof_b):.5f}\n")
                f.write(f"corr(up-LGBM, base18)={corr_a:.4f}\n")
                f.write(f"sweep_a: {sweep_a}\n")
                f.write(f"sweep_b: {sweep_b}\n")
                f.write(f"FINAL honest OOF combined={sc_f:.5f} "
                        f"BEATS {BAR18:.5f} at w_A={w_a:.2f} "
                        f"beta_B={beta_b:.2f}\n")
                f.write(f"spec: (1-w)*iter18 + w*upLGBM; HVC rows "
                        f"(1-beta)*blend + beta*specialist; "
                        f"seeds={MEMBER_SEEDS} feats=439 "
                        f"(spec top-{SPEC_TOPK})\n")
            st["path"] = path
            save_state(st)
            print(f"\nWINNER: {path} honest comb={sc_f:.5f} > "
                  f"bar {BAR18:.5f}", flush=True)
        else:
            print(f"\nNo winner: best combined blend {sc_f:.5f} does not "
                  f"beat iter18 bar {BAR18:.5f}. No file written.", flush=True)
    print("EXIT:0")


if __name__ == "__main__":
    main()
