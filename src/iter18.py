"""Iteration 18: RealMLP diversity member (research direction #2).

Trains RealMLP (pytabkit reference implementation, tuned defaults) on the
same 439-feature set as iter16 (399 base + 40 kept v6), with the SAME honest
customer-grouped validation (StratifiedGroupKFold on customer; never
row-wise CV). Blends RealMLP at 5-10% weight with iter16's 10-seed
LGBM+CatBoost blend (reconstructed from cached iter16 npz files; iter16's
10 seeds are NOT retrained).

Why RealMLP: iter17's asymmetric CatBoost hedge FAILED its gate (best
0.82550 vs bar 0.82557) because its OOF correlated 0.9961 with iter16 --
no diversity. RealMLP is a genuinely different inductive bias (TabArena
Elo: RealMLP Tuned+Ensemble 1489.9 vs LightGBM 1412.5, CatBoost 1398.1).

Gate: blended OOF must beat iter16's honest OOF comb 0.82557 on grouped
OOF. If it beats the bar -> submissions/iter18.csv. If not -> no file.

Protocol mirrors iter16/iter17 exactly: per-seed grouped TE, 5-fold grouped
CV, per-fold isotonic calibration, combined = 0.6*(1-LL) + 0.4*AUC.
Resume-safe via per-seed npz (iter18_realmlp_s{seed}.npz) +
iter18_state.json. NEVER submits (parent handles browser). Commit, don't push.
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iter16
from iter16 import SEEDS10, add_te_grouped, add_v6, build_399_honest, sgkf
from iter6 import EPS, ID_COL, OUT, combined
from iter8 import score_oof, CAT_ORDER
from iter17 import reconstruct_iter16_oof

BAR16 = 0.8255744471325221            # iter16 honest OOF comb (A_wg0.5_wl0.4)
RMLP_SEEDS = [42, 7]                  # 2 seeds; seed 42 first, 7 if time allows
W_GRID = [0.05, 0.10, 0.15]
STATE18 = os.path.join(OUT, "iter18_state.json")
_TORCH_THREADS_SET = False


def save_state(st):
    with open(STATE18, "w") as f:
        json.dump(st, f)


def load_state():
    if os.path.exists(STATE18):
        with open(STATE18) as f:
            return json.load(f)
    return {}


def prep_frame(df, feats):
    """Frame for pytabkit RealMLP: categoricals as 'category' dtype,
    numerics float32. Returns a copy restricted to feats."""
    X = df[feats].copy()
    for c in CAT_ORDER:
        if c in X.columns:
            X[c] = X[c].astype("category")
    for c in X.columns:
        if c not in CAT_ORDER:
            X[c] = X[c].astype(np.float32)
    return X


def make_realmlp(seed, n_epochs=None):
    """pytabkit RealMLP_TD_Classifier (tuned defaults), CPU.

    Deviations from pure TD, all principled for our metric (60% logloss):
    - val_metric_name='cross_entropy': best-epoch selection on logloss
      (TD default 'class_error' is wrong for a logloss-heavy metric).
    - batch_size='auto': authors' own heuristic (512 at 32k rows), halves
      wall-clock on CPU vs 256.
    - use_early_stopping=True (default patiences): stops at >2*best+20
      epochs; best-epoch checkpoint is still restored.
    """
    from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Classifier
    import torch
    global _TORCH_THREADS_SET
    if not _TORCH_THREADS_SET:
        # set_num_interop_threads errors if called twice in one process
        torch.set_num_threads(4)
        torch.set_num_interop_threads(2)
        _TORCH_THREADS_SET = True
    kw = dict(device="cpu", random_state=seed, n_cv=1, n_threads=4,
              verbosity=0, val_metric_name="cross_entropy",
              batch_size="auto", use_early_stopping=True)
    if n_epochs is not None:
        kw["n_epochs"] = n_epochs
    return RealMLP_TD_Classifier(**kw)


def run_rmlp_grouped(X, y, Xte, seed, groups, n_epochs=None):
    """Grouped 5-fold CV RealMLP + per-fold isotonic. The grouped val fold
    is passed as explicit validation for early stopping / best-epoch
    selection (no row-wise internal split). Per-fold checkpointing: after
    each fold, oof/test partials are saved so a relaunch resumes at the
    next fold. Returns oof, test."""
    partial_p = os.path.join(OUT, f"iter18_realmlp_s{seed}_partial.npz")
    if os.path.exists(partial_p):
        d = np.load(partial_p)
        oof = d["oof"].copy()
        test_sum = d["test_sum"].copy()
        folds_done = set(int(f) for f in d["folds_done"])
        print(f"seed {seed}: resuming from partial, folds done "
              f"{sorted(folds_done)}", flush=True)
    else:
        oof = np.zeros(len(X))
        test_sum = np.zeros(len(Xte))
        folds_done = set()
    for fold, (trn, val) in enumerate(sgkf(seed).split(X, y, groups)):
        if fold in folds_done:
            continue
        m = make_realmlp(seed, n_epochs=n_epochs)
        Xtr = prep_frame(X.iloc[trn], list(X.columns))
        Xva = prep_frame(X.iloc[val], list(X.columns))
        Xte_f = prep_frame(Xte, list(X.columns))
        # explicit grouped validation for early stopping / best epoch
        m.fit(Xtr, y[trn], X_val=Xva, y_val=y[val])
        pv = np.clip(m.predict_proba(Xva)[:, 1], EPS, 1 - EPS)
        pt = np.clip(m.predict_proba(Xte_f)[:, 1], EPS, 1 - EPS)
        iso = IsotonicRegression(out_of_bounds="clip")
        oof[val] = np.clip(iso.fit_transform(pv, y[val]), EPS, 1 - EPS)
        test_sum += np.clip(iso.transform(np.clip(pt, 0, 1)), EPS, 1 - EPS)
        folds_done.add(fold)
        np.savez(partial_p, oof=oof, test_sum=test_sum,
                 folds_done=np.array(sorted(folds_done)))
        print(f"  seed {seed} fold {fold}: comb="
              f"{score_oof(y[val], oof[val]):.5f} (checkpointed)",
              flush=True)
    test = test_sum / 5
    return oof, test


def train_rmlp_seed(seed, tr0, te0, y, feats, groups, n_epochs=None):
    p = os.path.join(OUT, f"iter18_realmlp_s{seed}.npz")
    if os.path.exists(p):
        d = np.load(p)
        print(f"seed {seed}: RealMLP loaded cached", flush=True)
        return d["oof"], d["test"]
    tr = tr0.copy()
    te = te0.copy()
    tr, te = add_te_grouped(tr, te, y, groups, seed=seed)
    print(f"seed {seed}: training grouped RealMLP (439 feats)...", flush=True)
    oof, test = run_rmlp_grouped(tr[feats], y, te[feats], seed, groups,
                                 n_epochs=n_epochs)
    np.savez(p, oof=oof, test=test)
    partial_p = os.path.join(OUT, f"iter18_realmlp_s{seed}_partial.npz")
    if os.path.exists(partial_p):
        os.remove(partial_p)
    print(f"seed {seed}: RealMLP done, comb={score_oof(y, oof):.5f} "
          f"oof_mean={oof.mean():.4f}", flush=True)
    return oof, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="train,blend",
                    help="comma-separated subset: train,blend")
    ap.add_argument("--seeds", default="42,7",
                    help="comma-separated RealMLP seeds to train")
    ap.add_argument("--n-epochs", type=int, default=None,
                    help="override RealMLP n_epochs (default: TD 256)")
    args = ap.parse_args()
    stages = set(args.stages.split(","))
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
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

    rmlp_oof = rmlp_test = None
    if "train" in stages:
        oofs_r, tests_r = [], []
        for s in seeds:
            oof_r, test_r = train_rmlp_seed(s, tr0, te0, y, feats_final,
                                            groups, n_epochs=args.n_epochs)
            oofs_r.append(oof_r)
            tests_r.append(test_r)
            st["rmlp_seeds_done"] = sorted(
                st.get("rmlp_seeds_done", []) + [s])
            save_state(st)
        rmlp_oof = np.clip(np.mean(oofs_r, axis=0), EPS, 1 - EPS)
        rmlp_test = np.clip(np.mean(tests_r, axis=0), EPS, 1 - EPS)
        sc_r = score_oof(y, rmlp_oof)
        print(f"\nRealMLP ({len(seeds)}-seed mean) OOF comb={sc_r:.5f} "
              f"oof_mean={rmlp_oof.mean():.4f} test_mean={rmlp_test.mean():.4f}",
              flush=True)
        st["rmlp_oof_comb"] = sc_r
        save_state(st)
    else:
        oofs_r, tests_r = [], []
        for s in seeds:
            d = np.load(os.path.join(OUT, f"iter18_realmlp_s{s}.npz"))
            oofs_r.append(d["oof"])
            tests_r.append(d["test"])
        rmlp_oof = np.clip(np.mean(oofs_r, axis=0), EPS, 1 - EPS)
        rmlp_test = np.clip(np.mean(tests_r, axis=0), EPS, 1 - EPS)

    if "blend" in stages:
        oof16, test16 = reconstruct_iter16_oof(y)
        corr = float(np.corrcoef(oof16, rmlp_oof)[0, 1])
        print(f"\nRealMLP vs iter16 OOF correlation: {corr:.4f} "
              f"(iter17 hedge was 0.9961)", flush=True)
        st["corr_rmlp_iter16"] = corr
        print("\nRealMLP-weight sweep (final = (1-w)*iter16 + w*RealMLP):",
              flush=True)
        best = None
        for w in W_GRID:
            oof_f = np.clip((1 - w) * oof16 + w * rmlp_oof, EPS, 1 - EPS)
            sc = score_oof(y, oof_f)
            print(f"  w={w:.2f}  comb={sc:.5f} "
                  f"({'BEATS' if sc > BAR16 else 'below'} bar {BAR16:.5f})",
                  flush=True)
            if best is None or sc > best[1]:
                best = (w, sc, oof_f)
        w_best, sc_best, oof_best = best
        st["sweep"] = {str(w): score_oof(
            y, np.clip((1 - w) * oof16 + w * rmlp_oof, EPS, 1 - EPS))
            for w in W_GRID}
        st["best_w"] = w_best
        st["best_comb"] = sc_best
        save_state(st)
        print(f"\nBest RealMLP weight w={w_best:.2f} comb={sc_best:.5f} "
              f"vs iter16 bar {BAR16:.5f}", flush=True)
        if sc_best > BAR16:
            test_f = np.clip((1 - w_best) * test16 + w_best * rmlp_test,
                             1e-6, 1 - 1e-6)
            path = os.path.join(OUT, "iter18.csv")
            pd.DataFrame({ID_COL: te_ids, "Target": test_f}).to_csv(
                path, index=False)
            print(f"Saved submissions/iter18.csv (test mean "
                  f"{test_f.mean():.4f}, honest OOF comb {sc_best:.5f})",
                  flush=True)
            with open(os.path.join(OUT, "iter18_results.txt"), "w") as f:
                f.write("ITERATION 18 results (HONEST customer-grouped CV)\n")
                f.write(f"iter16 bar: {BAR16:.5f}\n")
                f.write(f"RealMLP ({len(seeds)} seeds, pytabkit defaults): "
                        f"OOF comb={score_oof(y, rmlp_oof):.5f}\n")
                f.write(f"corr(RealMLP OOF, iter16 OOF)={corr:.4f}\n")
                f.write(f"FINAL honest OOF combined={sc_best:.5f} "
                        f"BEATS {BAR16:.5f} at w={w_best:.2f}\n")
                f.write(f"sweep: {st['sweep']}\n")
                f.write(f"spec: (1-w)*iter16_A_wg0.5_wl0.4 + w*RealMLP "
                        f"seeds={seeds} feats={len(feats_final)}\n")
            st["path"] = path
            save_state(st)
            print(f"\nWINNER: {path} honest comb={sc_best:.5f} > "
                  f"bar {BAR16:.5f}", flush=True)
        else:
            print(f"\nNo winner: best RealMLP blend {sc_best:.5f} does not "
                  f"beat iter16 bar {BAR16:.5f}. No file written.", flush=True)
    print("EXIT:0")


if __name__ == "__main__":
    main()
