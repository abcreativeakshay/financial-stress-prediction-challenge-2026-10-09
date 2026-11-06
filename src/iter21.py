"""Iteration 21: research directions #4 (monotone LightGBM) + #5 (top-K
pruning + TruncatedSVD-14), each under the NESTED protocol.

Context: iter18 honest OOF 0.82568, public 0.72354 (BEST public).
iter19 HVC specialist honest OOF 0.82769 but public 0.72209 -- BELOW
iter18. iter20 (specialist extension, nested-protocol verified) honest
OOF 0.82789, verification-half margin only +0.00004 -- submitted, public
0.72191 (2026-11-06), BELOW iter19 and iter18. LESSON: weights/blends
picked on the same OOF used for gating do NOT transfer (selection bias);
specialist-family OOF gains do not transfer to public at all. The
specialist direction is exhausted. Even if iter21 passes the nested gate
below, public transfer is uncertain -- the gate is the best evidence
available, not a promise.

Nested protocol (this script, same as iter20):
  * Split the 10,000 customers (grouped, stratified by customer positive
    count, seed 123) into a SELECTION half and a VERIFICATION half.
  * ALL picks -- monotone signs, K, blend weights -- are made on the
    SELECTION half only. The picked recipe is verified on the untouched
    VERIFICATION half against the iter20 recipe on the same rows.
  * Gate: a candidate must beat iter20's honest OOF 0.82789 on the FULL
    grouped OOF AND show a positive margin on the verification half.
    Otherwise NO file, report numbers.

Direction #4 (monotone LightGBM):
  * Signs derived empirically on the SELECTION half only: per feature,
    decile bins, mean target rate per decile, Spearman rho; +1 if
    rho >= +0.20 with matching first-vs-last decile direction, -1 if
    rho <= -0.20, else 0. Categoricals and TE features unconstrained.
    Agreement with the iter20-base OOF partial dependence is reported
    as a diagnostic (not used for the pick).
  * Train monotone LGBM (KITCHEN_SINK + monotone_constraints, method
    "advanced"), 5 seeds [42,7,2025,5,11], grouped 5-fold CV +
    per-fold isotonic, 439 feats. Blend with the iter20 incumbent OOF
    at w in [0.10, 0.20, 0.30], picked on the selection half.

Direction #5 (top-K pruning + TruncatedSVD-14):
  * Gain ranking fit on SELECTION-half rows only (439 feats, seed 42,
    800 rounds); top K=220 features kept.
  * TruncatedSVD-14 on standardized pruned numeric features as 14 extra
    inputs. SVD is fit INSIDE each CV fold (train-fold only) for OOF
    honesty; for test predictions the per-fold SVDs are averaged like
    the per-fold isotonic test predictions.
  * Retrain the blend on the pruned+SVD set: LGBM + CatBoost (4 seeds
    [42,7,2025,5] each), per-seed hybrid (wg=0.5, wl=0.4, as iter16),
    averaged over seeds. Blend with the iter20 incumbent OOF at w in
    [0.10, 0.20, 0.30], picked on the selection half.

Combination: if BOTH directions pass individually, a combined blend is
swept on the selection half and used ONLY if it beats both singles on
both the full OOF and the verification half.

Reuses cached npz: iter16/iter18 base seeds, RealMLP seeds, iter20
specialist seeds -- NONE retrained. Resume-safe via per-seed npz +
iter21_state.json. NEVER submits (parent handles browser). Commit,
don't push.

PATCH 2026-11-06 ~08:30 UTC (after 2 VM reboots killed the run twice
mid mono seed 5 with no checkpoint): all three grouped trainers are
now PER-FOLD checkpointed (iter21_*_partial.npz, same pattern as
iter18's RealMLP partials) with per-fold progress prints, and LGBM
rounds are capped at 2000 (finished mono seeds stopped at ~580-610,
so the cap is immaterial for normal folds but bounds pathological
ones -- seed 5 consumed 65+ min CPU without finishing in run 3).
The running process already loaded the old module, so this patch
only takes effect on the next relaunch.
PATCH 2, 2026-11-06 ~10:05 UTC: seed 5 fold 0 ran 70+ min / ~50
CPU-min in run 4 (PID 1682) without finishing one fold, vs ~5
CPU-min/fold for seeds 42/7/2025 -- ~9x anomaly, cause unknown
(memory pressure: 183MB free, iter22 contending). Killed PID 1682
and lowered the LGBM cap 2000 -> 900 (finished seeds stopped at
580-607, so normal folds are unaffected; a pathological fold now
costs at most ~1.5x a normal one). Per-fold checkpoints from PATCH 1
remain in effect.
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iter16
from iter16 import (CB_D8, SEEDS10, add_te_grouped, add_v6,
                    build_399_honest, sgkf, run_cb_grouped)
from iter6 import (KITCHEN_SINK, CAT_COLS, N_SPLITS, TARGET, ID_COL, DATA,
                   OUT, EPS, combined)
from iter8 import score_oof, CAT_ORDER
from iter12 import hybrid
from iter19 import hvc_masks, reconstruct_base18
from iter20 import nested_split, blend_hvc, SPLIT_SEED

BAR20 = 0.8278876                  # iter20 honest OOF comb (full)
BETA20 = 0.90                      # iter20 specialist beta on HVC rows
SPEC_K80_SEEDS = [42, 7, 2025, 5]  # iter20 top-80 specialist seeds
MONO_SEEDS = [42, 7, 2025, 5, 11]
PRUNE_SEEDS = [42, 7, 2025, 5]
TOPK = 220
N_SVD = 14
W_GRID = [0.10, 0.20, 0.30]
STATE21 = os.path.join(OUT, "iter21_state.json")


def save_state(st):
    with open(STATE21, "w") as f:
        json.dump(st, f)


def load_state():
    if os.path.exists(STATE21):
        with open(STATE21) as f:
            return json.load(f)
    return {}


def reconstruct_incumbent20(y, hvc_tr):
    """iter20 recipe OOF + test: base18 with HVC rows blended toward the
    top-80 4-seed LGBM specialist at beta=0.90."""
    oof18, test18, _, _ = reconstruct_base18(y)
    oofs = [np.load(os.path.join(OUT, f"iter20_spec_k80_s{s}.npz"))["oof"]
            for s in SPEC_K80_SEEDS]
    oof_k80_h = np.clip(np.mean(oofs, axis=0), EPS, 1 - EPS)
    spec_full = oof18.copy()
    spec_full[hvc_tr] = oof_k80_h
    oof20 = blend_hvc(oof18, spec_full, hvc_tr, BETA20)
    sc = score_oof(y, oof20)
    print(f"reconstructed iter20 incumbent OOF: comb={sc:.5f} "
          f"(expected {BAR20:.5f})", flush=True)
    assert abs(sc - BAR20) < 1e-4, f"iter20 reconstruction mismatch: {sc}"
    test20 = pd.read_csv(os.path.join(OUT, "iter20.csv"))["Target"].values
    print(f"iter20 test (from csv): mean={test20.mean():.4f} "
          f"n={len(test20)}", flush=True)
    return oof20, test20, oof18, test18


# ---------------------------------------------------------------- signs
def derive_mono_signs(tr_sel, y_sel, oof20_sel, feats, rho_thr=0.20):
    """Monotone signs from SELECTION-half rows only. Per feature: decile
    bins, mean target rate per decile, Spearman rho of (decile, rate).
    +1 if rho >= thr and last-decile rate > first-decile rate; -1 if
    rho <= -thr and last < first; else 0. Categoricals / TE excluded.
    Also reports agreement with the incumbent-OOF partial dependence
    (decile means of oof20) as a diagnostic only."""
    signs = {}
    agree, n_con = 0, 0
    for f in feats:
        if f in CAT_COLS or f.startswith("te_"):
            signs[f] = 0
            continue
        x = tr_sel[f].values
        if len(np.unique(x)) < 5:
            signs[f] = 0
            continue
        try:
            bins = pd.qcut(x, 10, labels=False, duplicates="drop")
        except ValueError:
            signs[f] = 0
            continue
        df = pd.DataFrame({"b": bins, "y": y_sel, "p": oof20_sel})
        rate = df.groupby("b")["y"].mean()
        if len(rate) < 5:
            signs[f] = 0
            continue
        rho = float(pd.Series(rate.values).corr(
            pd.Series(np.arange(len(rate))), method="spearman"))
        s = 0
        if rho >= rho_thr and rate.iloc[-1] > rate.iloc[0]:
            s = 1
        elif rho <= -rho_thr and rate.iloc[-1] < rate.iloc[0]:
            s = -1
        signs[f] = s
        if s != 0:
            n_con += 1
            prate = df.groupby("b")["p"].mean()
            rho_p = float(pd.Series(prate.values).corr(
                pd.Series(np.arange(len(prate))), method="spearman"))
            if (rho_p > 0) == (s > 0):
                agree += 1
    print(f"monotone signs: {n_con} constrained of {len(feats)} "
          f"({sum(1 for v in signs.values() if v > 0)} +, "
          f"{sum(1 for v in signs.values() if v < 0)} -); OOF-PD agrees "
          f"on {agree}/{n_con} constrained features", flush=True)
    return signs, n_con, agree


# ------------------------------------------------------- monotone LGBM
def run_lgbm_mono_grouped(X, y, Xte, params, groups, seed, mono):
    """Grouped 5-fold CV LGBM + per-fold isotonic, with monotone
    constraints (list aligned to X.columns). PER-FOLD checkpointed:
    a reboot loses at most one fold, not the whole seed. Rounds capped
    at 2000 (finished seeds stopped at ~580-610, so the cap is
    immaterial for normal folds but bounds pathological ones)."""
    tag = f"iter21_mono_lgbm_s{seed}"
    partial_p = os.path.join(OUT, f"{tag}_partial.npz")
    if os.path.exists(partial_p):
        d = np.load(partial_p)
        oof = d["oof"].copy()
        test_sum = d["test_sum"].copy()
        folds_done = set(int(f) for f in d["folds_done"])
        iters = [float(v) for v in d["iters"]]
        print(f"seed {seed}: resuming mono LGBM from partial, folds done "
              f"{sorted(folds_done)}", flush=True)
    else:
        oof = np.zeros(len(X))
        test_sum = np.zeros(len(Xte))
        folds_done = set()
        iters = []
    cat_here = [c for c in CAT_COLS if c in X.columns]
    for fold, (trn, val) in enumerate(sgkf(seed).split(X, y, groups)):
        if fold in folds_done:
            continue
        dtr = lgb.Dataset(X.iloc[trn], y[trn], categorical_feature=cat_here)
        dva = lgb.Dataset(X.iloc[val], y[val], categorical_feature=cat_here,
                          reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=900, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        pv = m.predict(X.iloc[val], num_iteration=m.best_iteration)
        pt = m.predict(Xte, num_iteration=m.best_iteration)
        iso = IsotonicRegression(out_of_bounds="clip")
        oof[val] = np.clip(iso.fit_transform(pv, y[val]), 1e-6, 1 - 1e-6)
        test_sum += np.clip(iso.transform(np.clip(pt, 0, 1)), 1e-6, 1 - 1e-6)
        iters.append(m.best_iteration)
        folds_done.add(fold)
        np.savez(partial_p, oof=oof, test_sum=test_sum,
                 folds_done=np.array(sorted(folds_done)),
                 iters=np.array(iters, dtype=float))
        print(f"  seed {seed} fold {fold}: best_iter={m.best_iteration} "
              f"(checkpointed)", flush=True)
    test = test_sum / N_SPLITS
    return oof, test, float(np.mean(iters))


def train_mono_seed(seed, tr0, te0, y, feats, groups, mono_list):
    p = os.path.join(OUT, f"iter21_mono_lgbm_s{seed}.npz")
    if os.path.exists(p):
        d = np.load(p)
        print(f"seed {seed}: monotone LGBM loaded cached", flush=True)
        return d["oof"], d["test"]
    tr = tr0.copy()
    te = te0.copy()
    tr, te = add_te_grouped(tr, te, y, groups, seed=seed)
    prm = dict(KITCHEN_SINK, seed=seed, monotone_constraints=mono_list,
               monotone_constraints_method="advanced")
    print(f"seed {seed}: training monotone LGBM "
          f"({sum(1 for v in mono_list if v != 0)} constrained feats)...",
          flush=True)
    oof, test, mi = run_lgbm_mono_grouped(tr[feats], y, te[feats], prm,
                                          groups, seed, mono_list)
    np.savez(p, oof=oof, test=test, iters=mi)
    partial_p = os.path.join(OUT, f"iter21_mono_lgbm_s{seed}_partial.npz")
    if os.path.exists(partial_p):
        os.remove(partial_p)
    print(f"seed {seed}: monotone LGBM done, mean_iter={mi:.0f}, "
          f"comb={score_oof(y, oof):.5f}", flush=True)
    return oof, test


# ------------------------------------------------------------ prune+SVD
def gain_rank_selection(tr_sel, y_sel, feats, topk):
    """Top-k features by gain from one LGBM fit on SELECTION-half rows
    only. Returns the ranked list (first topk = kept)."""
    dall = lgb.Dataset(tr_sel[feats], y_sel, categorical_feature=CAT_COLS)
    m = lgb.train(dict(KITCHEN_SINK, seed=42, num_threads=4), dall,
                  num_boost_round=800)
    gain = pd.Series(m.feature_importance(importance_type="gain"), index=feats)
    ranked = list(gain.sort_values(ascending=False).index)
    print(f"selection-half gain ranking: top-5 {ranked[:5]}; "
          f"kept top-{topk}, dropped {len(feats) - topk}", flush=True)
    return ranked


def svd_transform_fit(Xtr_num, Xva_num, Xte_num, seed):
    """Fit StandardScaler + TruncatedSVD on TRAIN rows only; transform
    val and test. Returns (tr, va, te) component DataFrames (14 cols)."""
    sc = StandardScaler()
    Ztr = sc.fit_transform(Xtr_num)
    svd = TruncatedSVD(n_components=N_SVD, random_state=seed)
    Ctr = svd.fit_transform(Ztr)
    Cva = svd.transform(sc.transform(Xva_num))
    Cte = svd.transform(sc.transform(Xte_num))
    cols = [f"svd_{i}" for i in range(N_SVD)]
    return (pd.DataFrame(Ctr, index=Xtr_num.index, columns=cols),
            pd.DataFrame(Cva, index=Xva_num.index, columns=cols),
            pd.DataFrame(Cte, index=Xte_num.index, columns=cols))


def run_lgbm_pruned_grouped(X, y, Xte, params, groups, seed, num_feats):
    """Grouped 5-fold CV LGBM + per-fold isotonic on pruned feats + 14
    SVD components fitted in-fold (train-fold only). PER-FOLD
    checkpointed (see run_lgbm_mono_grouped)."""
    tag = f"iter21_pruned_lgbm_s{seed}"
    partial_p = os.path.join(OUT, f"{tag}_partial.npz")
    if os.path.exists(partial_p):
        d = np.load(partial_p)
        oof = d["oof"].copy()
        test_sum = d["test_sum"].copy()
        folds_done = set(int(f) for f in d["folds_done"])
        iters = [float(v) for v in d["iters"]]
        print(f"seed {seed}: resuming pruned LGBM from partial, folds "
              f"done {sorted(folds_done)}", flush=True)
    else:
        oof = np.zeros(len(X))
        test_sum = np.zeros(len(Xte))
        folds_done = set()
        iters = []
    cat_here = [c for c in CAT_COLS if c in X.columns]
    for fold, (trn, val) in enumerate(sgkf(seed).split(X, y, groups)):
        if fold in folds_done:
            continue
        Ctr, Cva, Cte = svd_transform_fit(
            X.iloc[trn][num_feats], X.iloc[val][num_feats],
            Xte[num_feats], seed)
        Xtr_f = pd.concat([X.iloc[trn].reset_index(drop=True),
                           Ctr.reset_index(drop=True)], axis=1)
        Xva_f = pd.concat([X.iloc[val].reset_index(drop=True),
                           Cva.reset_index(drop=True)], axis=1)
        Xte_f = pd.concat([Xte.reset_index(drop=True),
                           Cte.reset_index(drop=True)], axis=1)
        dtr = lgb.Dataset(Xtr_f, y[trn], categorical_feature=cat_here)
        dva = lgb.Dataset(Xva_f, y[val], categorical_feature=cat_here,
                          reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=900, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        pv = m.predict(Xva_f, num_iteration=m.best_iteration)
        pt = m.predict(Xte_f, num_iteration=m.best_iteration)
        iso = IsotonicRegression(out_of_bounds="clip")
        oof[val] = np.clip(iso.fit_transform(pv, y[val]), 1e-6, 1 - 1e-6)
        test_sum += np.clip(iso.transform(np.clip(pt, 0, 1)), 1e-6, 1 - 1e-6)
        iters.append(m.best_iteration)
        folds_done.add(fold)
        np.savez(partial_p, oof=oof, test_sum=test_sum,
                 folds_done=np.array(sorted(folds_done)),
                 iters=np.array(iters, dtype=float))
        print(f"  seed {seed} fold {fold}: best_iter={m.best_iteration} "
              f"(checkpointed)", flush=True)
    test = test_sum / N_SPLITS
    return oof, test, float(np.mean(iters))


def run_cb_pruned_grouped(X, y, Xte, cfg, seed, groups, num_feats):
    """Grouped 5-fold CV CatBoost(d8) + per-fold isotonic on pruned feats
    + 14 SVD components fitted in-fold (train-fold only). PER-FOLD
    checkpointed (see run_lgbm_mono_grouped)."""
    tag = f"iter21_pruned_cb_s{seed}"
    partial_p = os.path.join(OUT, f"{tag}_partial.npz")
    if os.path.exists(partial_p):
        d = np.load(partial_p)
        oof = d["oof"].copy()
        test_sum = d["test_sum"].copy()
        folds_done = set(int(f) for f in d["folds_done"])
        iters = [float(v) for v in d["iters"]]
        print(f"seed {seed}: resuming pruned CB from partial, folds done "
              f"{sorted(folds_done)}", flush=True)
    else:
        oof = np.zeros(len(X))
        test_sum = np.zeros(len(Xte))
        folds_done = set()
        iters = []
    for fold, (trn, val) in enumerate(sgkf(seed).split(X, y, groups)):
        if fold in folds_done:
            continue
        Ctr, Cva, Cte = svd_transform_fit(
            X.iloc[trn][num_feats], X.iloc[val][num_feats],
            Xte[num_feats], seed)
        Xtr_f = pd.concat([X.iloc[trn].reset_index(drop=True),
                           Ctr.reset_index(drop=True)], axis=1)
        Xva_f = pd.concat([X.iloc[val].reset_index(drop=True),
                           Cva.reset_index(drop=True)], axis=1)
        Xte_f = pd.concat([Xte.reset_index(drop=True),
                           Cte.reset_index(drop=True)], axis=1)
        for c in CAT_ORDER:
            if c in Xtr_f.columns:
                Xtr_f[c] = Xtr_f[c].astype(str)
                Xva_f[c] = Xva_f[c].astype(str)
                Xte_f[c] = Xte_f[c].astype(str)
        m = CatBoostClassifier(iterations=cfg["iterations"],
                               learning_rate=cfg["learning_rate"],
                               depth=cfg["depth"],
                               l2_leaf_reg=cfg["l2_leaf_reg"],
                               loss_function="Logloss", eval_metric="Logloss",
                               random_seed=seed, early_stopping_rounds=100,
                               use_best_model=True, verbose=False,
                               allow_writing_files=False, thread_count=4)
        m.fit(Xtr_f, y[trn], eval_set=(Xva_f, y[val]),
              cat_features=[c for c in CAT_ORDER if c in Xtr_f.columns])
        pv = m.predict_proba(Xva_f)[:, 1]
        pt = m.predict_proba(Xte_f)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        oof[val] = np.clip(iso.fit_transform(pv, y[val]), EPS, 1 - EPS)
        test_sum += np.clip(iso.transform(np.clip(pt, 0, 1)), EPS, 1 - EPS)
        iters.append(m.get_best_iteration())
        folds_done.add(fold)
        np.savez(partial_p, oof=oof, test_sum=test_sum,
                 folds_done=np.array(sorted(folds_done)),
                 iters=np.array(iters, dtype=float))
        print(f"  seed {seed} fold {fold}: best_iter={m.get_best_iteration()} "
              f"(checkpointed)", flush=True)
    test = test_sum / N_SPLITS
    return oof, test, float(np.mean(iters))


def train_pruned_lgbm_seed(seed, tr0, te0, y, feats_k, groups, num_feats):
    p = os.path.join(OUT, f"iter21_pruned_lgbm_s{seed}.npz")
    if os.path.exists(p):
        d = np.load(p)
        print(f"seed {seed}: pruned LGBM loaded cached", flush=True)
        return d["oof"], d["test"]
    tr = tr0.copy()
    te = te0.copy()
    tr, te = add_te_grouped(tr, te, y, groups, seed=seed)
    print(f"seed {seed}: training pruned+SVD LGBM "
          f"({len(feats_k)} + {N_SVD} feats)...", flush=True)
    oof, test, mi = run_lgbm_pruned_grouped(
        tr[feats_k], y, te[feats_k], dict(KITCHEN_SINK, seed=seed),
        groups, seed, num_feats)
    np.savez(p, oof=oof, test=test, iters=mi)
    partial_p = os.path.join(OUT, f"iter21_pruned_lgbm_s{seed}_partial.npz")
    if os.path.exists(partial_p):
        os.remove(partial_p)
    print(f"seed {seed}: pruned LGBM done, mean_iter={mi:.0f}, "
          f"comb={score_oof(y, oof):.5f}", flush=True)
    return oof, test


def train_pruned_cb_seed(seed, tr0, te0, y, feats_k, groups, num_feats):
    p = os.path.join(OUT, f"iter21_pruned_cb_s{seed}.npz")
    if os.path.exists(p):
        d = np.load(p)
        print(f"seed {seed}: pruned CatBoost loaded cached", flush=True)
        return d["oof"], d["test"]
    tr = tr0.copy()
    te = te0.copy()
    tr, te = add_te_grouped(tr, te, y, groups, seed=seed)
    print(f"seed {seed}: training pruned+SVD CatBoost "
          f"({len(feats_k)} + {N_SVD} feats)...", flush=True)
    oof, test, mi = run_cb_pruned_grouped(
        tr[feats_k], y, te[feats_k], CB_D8, seed, groups, num_feats)
    np.savez(p, oof=oof, test=test, iters=mi)
    partial_p = os.path.join(OUT, f"iter21_pruned_cb_s{seed}_partial.npz")
    if os.path.exists(partial_p):
        os.remove(partial_p)
    print(f"seed {seed}: pruned CB done, mean_iter={mi:.0f}, "
          f"comb={score_oof(y, oof):.5f}", flush=True)
    return oof, test


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="signs,mono,prune,blend,combine",
                    help="comma-separated subset: "
                         "signs,mono,prune,blend,combine")
    args = ap.parse_args()
    stages = set(args.stages.split(","))
    os.makedirs(OUT, exist_ok=True)
    st = load_state()

    st16 = json.load(open(os.path.join(OUT, "iter16_state.json")))
    kept_v6 = st16["step1_kept"]
    assert len(kept_v6) == 40

    tr399, te399, y, feats399, groups = build_399_honest(seed=42)
    te_ids = te399[ID_COL].values
    feats_final = feats399 + kept_v6
    tr0, te0, _ = add_v6(tr399.copy(), te399.copy())
    hvc_tr, hvc_te = hvc_masks(tr399, te399)
    print(f"feats_final={len(feats_final)}; HVC rows: train "
          f"{hvc_tr.sum()} / {len(y)}, test {hvc_te.sum()} / {len(te399)}",
          flush=True)

    oof20, test20, oof18, test18 = reconstruct_incumbent20(y, hvc_tr)
    sel, ver = nested_split(groups, y)
    st["nested_split_seed"] = SPLIT_SEED
    save_state(st)

    def comb_on(mask, p):
        return score_oof(y[mask], p[mask])

    ver_20 = comb_on(ver, oof20)
    sel_20 = comb_on(sel, oof20)
    full_20 = score_oof(y, oof20)
    print(f"\niter20 incumbent: selection={sel_20:.5f} "
          f"verification={ver_20:.5f} full={full_20:.5f}", flush=True)
    st["incumbent20"] = {"sel": sel_20, "ver": ver_20, "full": full_20}
    save_state(st)

    # ---- stage signs: monotone signs on the selection half only ----
    mono_list = None
    if "signs" in stages:
        tr42, _ = add_te_grouped(tr0.copy(), te0.copy(), y, groups, seed=42)
        signs, n_con, agree = derive_mono_signs(
            tr42[sel], y[sel], oof20[sel], feats_final)
        mono_list = [signs[f] for f in feats_final]
        st["mono_signs"] = {"n_constrained": n_con, "oof_pd_agree": agree,
                            "n_pos": sum(1 for v in mono_list if v > 0),
                            "n_neg": sum(1 for v in mono_list if v < 0),
                            "constrained_feats":
                                [f for f, v in zip(feats_final, mono_list)
                                 if v != 0]}
        save_state(st)
    else:
        s = st["mono_signs"]
        mono_list = [1 if f in s["constrained_feats"] and True else 0
                     for f in feats_final]
        # rebuild exact signs from the stored constrained list + rho sign
        # (signs were +1/-1; stored list alone loses the sign, so reload
        # from the saved full map when present)
        if "sign_map" in s:
            mono_list = [s["sign_map"].get(f, 0) for f in feats_final]
        else:
            raise RuntimeError("signs stage not run and no sign_map in "
                               "state -- rerun with --stages signs,...")
    if "signs" in stages:
        # persist the full sign map for resume-safety
        st["mono_signs"]["sign_map"] = {
            f: v for f, v in zip(feats_final, mono_list)}
        save_state(st)

    # ---- stage mono: 5-seed monotone LGBM ----
    oof_mono = test_mono = None
    if "mono" in stages:
        oofs, tests = [], []
        for s in MONO_SEEDS:
            oof_s, test_s = train_mono_seed(s, tr0, te0, y, feats_final,
                                            groups, mono_list)
            oofs.append(oof_s)
            tests.append(test_s)
            st["mono_seeds_done"] = sorted(
                st.get("mono_seeds_done", []) + [s])
            save_state(st)
        oof_mono = np.clip(np.mean(oofs, axis=0), EPS, 1 - EPS)
        test_mono = np.clip(np.mean(tests, axis=0), EPS, 1 - EPS)
        sc_m = score_oof(y, oof_mono)
        corr_m = float(np.corrcoef(oof20, oof_mono)[0, 1])
        print(f"\nMonotone LGBM ({len(MONO_SEEDS)}-seed mean) OOF "
              f"comb={sc_m:.5f} corr vs incumbent={corr_m:.4f}", flush=True)
        st["mono_oof_comb"] = sc_m
        st["corr_mono_incumbent"] = corr_m
        save_state(st)
    else:
        oofs = [np.load(os.path.join(
            OUT, f"iter21_mono_lgbm_s{s}.npz")) for s in MONO_SEEDS]
        oof_mono = np.clip(np.mean([d["oof"] for d in oofs], axis=0),
                           EPS, 1 - EPS)
        test_mono = np.clip(np.mean([d["test"] for d in oofs], axis=0),
                            EPS, 1 - EPS)

    # ---- stage prune: top-K + SVD-14 retrained blend ----
    oof_pr = test_pr = None
    if "prune" in stages:
        # gain ranking on SELECTION-half rows only
        tr42, _ = add_te_grouped(tr0.copy(), te0.copy(), y, groups, seed=42)
        ranked = gain_rank_selection(tr42[sel], y[sel], feats_final, TOPK)
        feats_k = ranked[:TOPK]
        st["prune_feats_k"] = feats_k
        save_state(st)
        num_feats = [f for f in feats_k if f not in CAT_COLS]
        print(f"pruned set: {len(feats_k)} feats "
              f"({len(num_feats)} numeric into SVD)", flush=True)
        oofs_l, tests_l, oofs_c, tests_c = [], [], [], []
        for s in PRUNE_SEEDS:
            oof_l, test_l = train_pruned_lgbm_seed(
                s, tr0, te0, y, feats_k, groups, num_feats)
            oof_c, test_c = train_pruned_cb_seed(
                s, tr0, te0, y, feats_k, groups, num_feats)
            oofs_l.append(oof_l)
            tests_l.append(test_l)
            oofs_c.append(oof_c)
            tests_c.append(test_c)
            st["prune_seeds_done"] = sorted(
                st.get("prune_seeds_done", []) + [s])
            save_state(st)
        hyb_o = [hybrid(oofs_l[i], oofs_c[i], 0.5, 0.4)
                 for i in range(len(PRUNE_SEEDS))]
        hyb_t = [hybrid(tests_l[i], tests_c[i], 0.5, 0.4)
                 for i in range(len(PRUNE_SEEDS))]
        oof_pr = np.clip(np.mean(hyb_o, axis=0), EPS, 1 - EPS)
        test_pr = np.clip(np.mean(hyb_t, axis=0), EPS, 1 - EPS)
        sc_p = score_oof(y, oof_pr)
        corr_p = float(np.corrcoef(oof20, oof_pr)[0, 1])
        print(f"\nPruned+SVD hybrid ({len(PRUNE_SEEDS)} seeds) OOF "
              f"comb={sc_p:.5f} corr vs incumbent={corr_p:.4f}", flush=True)
        st["pruned_oof_comb"] = sc_p
        st["corr_pruned_incumbent"] = corr_p
        save_state(st)
    else:
        oofs_l = [np.load(os.path.join(
            OUT, f"iter21_pruned_lgbm_s{s}.npz")) for s in PRUNE_SEEDS]
        oofs_c = [np.load(os.path.join(
            OUT, f"iter21_pruned_cb_s{s}.npz")) for s in PRUNE_SEEDS]
        hyb_o = [hybrid(oofs_l[i]["oof"], oofs_c[i]["oof"], 0.5, 0.4)
                 for i in range(len(PRUNE_SEEDS))]
        hyb_t = [hybrid(oofs_l[i]["test"], oofs_c[i]["test"], 0.5, 0.4)
                 for i in range(len(PRUNE_SEEDS))]
        oof_pr = np.clip(np.mean(hyb_o, axis=0), EPS, 1 - EPS)
        test_pr = np.clip(np.mean(hyb_t, axis=0), EPS, 1 - EPS)

    # ---- stage blend: nested selection + verification + gate ----
    winner = None
    if "blend" in stages:
        print("\n--- nested SELECTION half: sweep blend weights "
              "(picks made here only) ---", flush=True)
        cands = []  # (label, oof_member, test_member)
        for w in W_GRID:
            cands.append((f"mono w={w:.2f}", oof_mono, test_mono, w))
        for w in W_GRID:
            cands.append((f"pruned w={w:.2f}", oof_pr, test_pr, w))
        sel_results = []
        for label, oof_m, _, w in cands:
            p = np.clip((1 - w) * oof20 + w * oof_m, EPS, 1 - EPS)
            sc_sel = comb_on(sel, p)
            sel_results.append((sc_sel, label, oof_m, w))
            print(f"  {label} + incumbent: selection comb={sc_sel:.5f}",
                  flush=True)
        # diagnostics (not picks): standalone members + mono vs iter18 base
        print(f"  [diag] mono standalone: selection="
              f"{comb_on(sel, oof_mono):.5f} full={score_oof(y, oof_mono):.5f}",
              flush=True)
        print(f"  [diag] pruned standalone: selection="
              f"{comb_on(sel, oof_pr):.5f} full={score_oof(y, oof_pr):.5f}",
              flush=True)
        p18 = np.clip(0.8 * oof18 + 0.2 * oof_mono, EPS, 1 - EPS)
        print(f"  [diag] mono w=0.20 vs iter18 base: selection="
              f"{comb_on(sel, p18):.5f} full={score_oof(y, p18):.5f}",
              flush=True)
        st["diag"] = {
            "mono_standalone_full": score_oof(y, oof_mono),
            "pruned_standalone_full": score_oof(y, oof_pr),
            "mono_vs_iter18_w02_full": score_oof(y, p18)}
        sel_results.sort(key=lambda t: -t[0])
        # best per family (a family passes only if its best pick passes)
        family_best = {}
        for sc_sel, label, oof_m, w in sel_results:
            fam = label.split()[0]
            if fam not in family_best:
                family_best[fam] = (sc_sel, label, oof_m, w)
        print("\n--- nested VERIFICATION half: each family's best pick "
              "vs iter20 incumbent ---", flush=True)
        passed = {}
        for fam, (sc_sel, label, oof_m, w) in family_best.items():
            p = np.clip((1 - w) * oof20 + w * oof_m, EPS, 1 - EPS)
            ver_p = comb_on(ver, p)
            full_p = score_oof(y, p)
            gate_ver = ver_p > ver_20
            gate_full = full_p > BAR20
            print(f"  {label}: selection={sc_sel:.5f} "
                  f"verification={ver_p:.5f} (incumbent {ver_20:.5f}, "
                  f"delta {ver_p - ver_20:+.5f}) full={full_p:.5f} "
                  f"(bar {BAR20:.5f}, delta {full_p - BAR20:+.5f}) "
                  f"GATE={'PASS' if gate_ver and gate_full else 'FAIL'}",
                  flush=True)
            passed[fam] = {"label": label, "w": w, "sel": sc_sel,
                           "ver": ver_p, "full": full_p,
                           "ver_delta": ver_p - ver_20,
                           "full_delta": full_p - BAR20,
                           "pass": bool(gate_ver and gate_full)}
        st["families"] = passed
        save_state(st)
        # winner = best passing family by full OOF
        passing = [(v["full"], k, v) for k, v in passed.items() if v["pass"]]
        if passing:
            passing.sort(key=lambda t: -t[0])
            _, fam, v = passing[0]
            oof_m = oof_mono if fam == "mono" else oof_pr
            test_m = test_mono if fam == "mono" else test_pr
            w = v["w"]
            test_f = np.clip((1 - w) * test20 + w * test_m, 1e-6, 1 - 1e-6)
            assert len(test_f) == 30000 and not np.isnan(test_f).any()
            path = os.path.join(OUT, "iter21.csv")
            pd.DataFrame({ID_COL: te_ids, "Target": test_f}).to_csv(
                path, index=False)
            print(f"\nSaved submissions/iter21.csv from {v['label']} "
                  f"(test mean {test_f.mean():.4f}, full honest OOF comb "
                  f"{v['full']:.5f})", flush=True)
            winner = {"label": v["label"], "family": fam, "w": w,
                      "full_comb": v["full"], "ver_comb": v["ver"],
                      "path": path}
            st["winner"] = winner
            save_state(st)
        else:
            print("\nNo family passes the nested gate: no file written.",
                  flush=True)

    # ---- stage combine: only if BOTH families passed individually ----
    if "combine" in stages and winner is not None:
        fam = st.get("families", {})
        if fam.get("mono", {}).get("pass") and fam.get("pruned", {}).get("pass"):
            print("\n--- stage combine: both families passed; sweep "
                  "combined blend on the selection half ---", flush=True)
            wm_best = fam["mono"]["w"]
            wp_best = fam["pruned"]["w"]
            grid = [(wm_best, wp_best),
                    (0.10, 0.10), (0.20, 0.10), (0.10, 0.20),
                    (0.20, 0.20), (0.15, 0.15)]
            best_c = None
            for wm, wp in grid:
                p = np.clip(oof20 + wm * (oof_mono - oof20)
                            + wp * (oof_pr - oof20), EPS, 1 - EPS)
                sc_sel = comb_on(sel, p)
                print(f"  combine w_mono={wm:.2f} w_pruned={wp:.2f}: "
                      f"selection comb={sc_sel:.5f}", flush=True)
                if best_c is None or sc_sel > best_c[0]:
                    best_c = (sc_sel, wm, wp)
            _, wm, wp = best_c
            p = np.clip(oof20 + wm * (oof_mono - oof20)
                        + wp * (oof_pr - oof20), EPS, 1 - EPS)
            ver_c = comb_on(ver, p)
            full_c = score_oof(y, p)
            print(f"combined pick w_mono={wm:.2f} w_pruned={wp:.2f}: "
                  f"verification={ver_c:.5f} (winner "
                  f"{winner['ver_comb']:.5f}) full={full_c:.5f} (winner "
                  f"{winner['full_comb']:.5f})", flush=True)
            if ver_c > winner["ver_comb"] and full_c > winner["full_comb"]:
                test_f = np.clip(test20 + wm * (test_mono - test20)
                                 + wp * (test_pr - test20), 1e-6, 1 - 1e-6)
                assert len(test_f) == 30000 and not np.isnan(test_f).any()
                path = os.path.join(OUT, "iter21.csv")
                pd.DataFrame({ID_COL: te_ids, "Target": test_f}).to_csv(
                    path, index=False)
                print(f"combined OVERWRITES iter21.csv "
                      f"(test mean {test_f.mean():.4f}, full honest OOF "
                      f"comb {full_c:.5f})", flush=True)
                winner = {"label": f"combined w_mono={wm:.2f} "
                                   f"w_pruned={wp:.2f}", "family": "combined",
                          "w": [wm, wp], "full_comb": full_c,
                          "ver_comb": ver_c, "path": path}
                st["winner"] = winner
                save_state(st)
            else:
                print("combined does not beat the single-direction winner "
                      "on both halves; iter21.csv unchanged.", flush=True)
            st["combine"] = {"w_mono": wm, "w_pruned": wp,
                             "full_comb": full_c, "ver_comb": ver_c}
            save_state(st)
        else:
            print("\ncombine skipped: not both families passed.",
                  flush=True)

    # ---- results file ----
    with open(os.path.join(OUT, "iter21_results.txt"), "w") as f:
        f.write("ITERATION 21 results (HONEST customer-grouped CV, "
                "NESTED selection/verification protocol)\n")
        f.write(f"iter20 bar (full OOF): {BAR20:.5f}; iter20 public "
                f"0.72191 BELOW iter18 0.72354 -- specialist OOF gains "
                f"do not transfer; gate is evidence, not a promise.\n")
        f.write(f"incumbent20: sel={sel_20:.5f} ver={ver_20:.5f} "
                f"full={full_20:.5f}\n")
        f.write(f"mono standalone full={score_oof(y, oof_mono):.5f} "
                f"corr vs incumbent="
                f"{float(np.corrcoef(oof20, oof_mono)[0, 1]):.4f}\n")
        f.write(f"pruned+SVD standalone full={score_oof(y, oof_pr):.5f} "
                f"corr vs incumbent="
                f"{float(np.corrcoef(oof20, oof_pr)[0, 1]):.4f}\n")
        f.write(f"state: {json.dumps(st, default=str)}\n")
        if winner:
            f.write(f"WINNER: {winner['label']} full honest OOF comb="
                    f"{winner['full_comb']:.5f} (bar {BAR20:.5f}) "
                    f"verification comb={winner['ver_comb']:.5f} "
                    f"(incumbent {ver_20:.5f}) path={winner['path']}\n")
        else:
            f.write("NO WINNER under the nested gate; no file written.\n")
    print("EXIT:0")


if __name__ == "__main__":
    main()
