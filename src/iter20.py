"""Iteration 20: extend the HVC specialist that (apparently) worked in iter19.

CORRECTION vs iter19 (2026-11-06 ~08:09 IST): iter19 scored public 0.72209
(ID FSERDUfN) -- BELOW iter18's 0.72354 -- despite honest OOF +0.00201
(0.82769 vs 0.82568). The OOF gain did not transfer. Likely cause: beta=0.90
was swept on the SAME grouped OOF used for gating (selection bias inflates
the estimate), and/or public test distribution differs from grouped OOF.

So iter20 does NOT pick blend weights/beta by sweeping on the gating OOF.
Nested protocol (this script, stage "blend"):
  * Split the 10,000 customers (grouped, stratified by customer positive
    count) into a SELECTION half and a VERIFICATION half.
  * Sweep beta / specialist-mix weights using ONLY the selection half's OOF.
  * Verify the picked recipe on the VERIFICATION half against the iter19
    recipe (beta=0.90, 4-seed LGBM spec) evaluated on the SAME verification
    half. iter19's beta was selected on the full OOF (which includes the
    verification half), so this comparison is conservative for iter20.
  * Gate: write iter20.csv ONLY if (a) picked recipe beats the iter19
    recipe on the verification half AND (b) full-OOF comb with the picked
    recipe beats iter19's 0.82769. Otherwise NO file, report numbers.

Work, in order:
  1. spec-more: extend the LGBM HVC specialist from 4 seeds [42,7,2025,5]
     to 10 seeds (+[11,777,99,1234,2024,31337]); same top-120 gain feats
     from iter19_state.json, same protocol. Re-sweep beta 0.80-1.00.
  2. spec-cb: a CatBoost HVC specialist (same top-120 feats, grouped CV
     within HVC customers, per-fold isotonic, CB_D8, no spw); blend with
     the LGBM specialist on HVC rows if it adds diversity -- check corr,
     must be well below 0.99 or it is the same no-diversity trap as the
     iter17 hedge (0.9961) and iter19 up-LGBM (0.9940).
  3. topk (only after blend; ~20-40 min): wider feature-count tune for the
     LGBM specialist (top-80 vs top-120 vs top-200), picked on the
     selection half and verified on the verification half under the same
     protocol. May only overwrite iter20.csv if it beats the blend-stage
     winner under the nested gate.

Same honest customer-grouped validation throughout (StratifiedGroupKFold
on customer; never row-wise CV). Reuses cached npz: iter19 spec seeds,
iter16/iter18 base seeds, RealMLP seeds -- NONE of those are retrained.
HVC upweighting (#3a) is DEAD -- do not retry. iter17 hedge is DEAD.

Gate: final blend must beat iter19's honest OOF 0.82769 (full OOF) AND
beat the iter19 recipe on the verification half. NEVER submits (parent
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
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iter16
from iter16 import (CB_D8, SEEDS10, add_te_grouped, add_v6,
                    build_399_honest, sgkf)
from iter6 import (KITCHEN_SINK, CAT_COLS, N_SPLITS, TARGET, ID_COL, DATA,
                   OUT, EPS, combined)
from iter8 import score_oof
from iter19 import hvc_masks, run_spec_grouped, reconstruct_base18
from catboost import CatBoostClassifier

BAR19 = 0.82769                    # iter19 honest OOF comb (beta=0.90)
BAR19_FULL = 0.827685              # iter19 full-OOF comb, recomputed below
SPEC_SEEDS4 = [42, 7, 2025, 5]     # iter19 LGBM specialist seeds
SPEC_SEEDS10 = [42, 7, 2025, 5, 11, 777, 99, 1234, 2024, 31337]
CB_SEEDS = [42, 7, 2025, 5]        # CatBoost specialist seeds (match LGBM)
BETA_GRID = [0.80, 0.85, 0.90, 0.95, 1.00]
MIX_GRID = [0.0, 0.25, 0.50, 0.75, 1.00]   # LGBM weight in specialist mix
TOPK_VALUES = [80, 200]            # extra feature counts (120 = baseline)
SPEC_TOPK = 120
STATE20 = os.path.join(OUT, "iter20_state.json")
SPLIT_SEED = 123                   # customer-half split seed (nested protocol)


def save_state(st):
    with open(STATE20, "w") as f:
        json.dump(st, f)


def load_state():
    if os.path.exists(STATE20):
        with open(STATE20) as f:
            return json.load(f)
    return {}


def load_spec_npz(seed, topk=SPEC_TOPK):
    """Load one LGBM specialist seed: iter20 file first, then iter19 file
    (only for top-120, the original 4 seeds). Returns dict or None."""
    if topk == SPEC_TOPK:
        for p in (os.path.join(OUT, f"iter20_spec_s{seed}.npz"),
                  os.path.join(OUT, f"iter19_spec_s{seed}.npz")):
            if os.path.exists(p):
                return dict(np.load(p))
    else:
        p = os.path.join(OUT, f"iter20_spec_k{topk}_s{seed}.npz")
        if os.path.exists(p):
            return dict(np.load(p))
    return None


def train_spec_seed_lgbm(seed, tr0, te0, y, feats, groups, hvc_tr, hvc_te,
                         spec_feats, topk=SPEC_TOPK):
    tag = f"iter20_spec_s{seed}" if topk == SPEC_TOPK else \
        f"iter20_spec_k{topk}_s{seed}"
    p = os.path.join(OUT, f"{tag}.npz")
    if os.path.exists(p):
        d = np.load(p)
        print(f"seed {seed} k={topk}: specialist LGBM loaded cached",
              flush=True)
        return d["oof"], d["test"]
    tr = tr0.copy()
    te = te0.copy()
    tr, te = add_te_grouped(tr, te, y, groups, seed=seed)
    Xh = tr.loc[hvc_tr, spec_feats]
    yh = y[hvc_tr]
    gh = groups[hvc_tr]
    Xte_h = te.loc[hvc_te, spec_feats]
    print(f"seed {seed} k={topk}: training HVC specialist LGBM "
          f"({len(spec_feats)} feats, {hvc_tr.sum()} rows)...", flush=True)
    oof_h, test_h, mi = run_spec_grouped(
        Xh, yh, Xte_h, dict(KITCHEN_SINK, seed=seed, num_leaves=15), gh, seed)
    np.savez(p, oof=oof_h, test=test_h, iters=mi)
    print(f"seed {seed} k={topk}: specialist done, mean_iter={mi:.0f}, "
          f"HVC comb={score_oof(yh, oof_h):.5f}", flush=True)
    return oof_h, test_h


def run_spec_cb_grouped(X, y, Xte, cfg, groups, seed):
    """Grouped 5-fold CV CatBoost on HVC rows only + per-fold isotonic.
    All specialist features are numeric (no categoricals in top-120)."""
    oof = np.zeros(len(X))
    test = np.zeros(len(Xte))
    iters = []
    for trn, val in StratifiedGroupKFold(
            n_splits=N_SPLITS, shuffle=True,
            random_state=seed).split(X, y, groups):
        m = CatBoostClassifier(iterations=cfg["iterations"],
                               learning_rate=cfg["learning_rate"],
                               depth=cfg["depth"],
                               l2_leaf_reg=cfg["l2_leaf_reg"],
                               loss_function="Logloss", eval_metric="Logloss",
                               random_seed=seed, early_stopping_rounds=100,
                               use_best_model=True, verbose=False,
                               allow_writing_files=False, thread_count=4)
        m.fit(X.iloc[trn], y[trn], eval_set=(X.iloc[val], y[val]))
        pv = m.predict_proba(X.iloc[val])[:, 1]
        pt = m.predict_proba(Xte)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        oof[val] = np.clip(iso.fit_transform(pv, y[val]), EPS, 1 - EPS)
        test += np.clip(iso.transform(np.clip(pt, 0, 1)), EPS, 1 - EPS) \
            / N_SPLITS
        iters.append(m.get_best_iteration())
    return oof, test, float(np.mean(iters))


def train_spec_seed_cb(seed, tr0, te0, y, feats, groups, hvc_tr, hvc_te,
                       spec_feats):
    p = os.path.join(OUT, f"iter20_spec_cb_s{seed}.npz")
    if os.path.exists(p):
        d = np.load(p)
        print(f"seed {seed}: specialist CatBoost loaded cached", flush=True)
        return d["oof"], d["test"]
    tr = tr0.copy()
    te = te0.copy()
    tr, te = add_te_grouped(tr, te, y, groups, seed=seed)
    Xh = tr.loc[hvc_tr, spec_feats]
    yh = y[hvc_tr]
    gh = groups[hvc_tr]
    Xte_h = te.loc[hvc_te, spec_feats]
    print(f"seed {seed}: training HVC specialist CatBoost "
          f"({len(spec_feats)} feats, {hvc_tr.sum()} rows)...", flush=True)
    oof_h, test_h, mi = run_spec_cb_grouped(Xh, yh, Xte_h, CB_D8, gh, seed)
    np.savez(p, oof=oof_h, test=test_h, iters=mi)
    print(f"seed {seed}: specialist CB done, mean_iter={mi:.0f}, "
          f"HVC comb={score_oof(yh, oof_h):.5f}", flush=True)
    return oof_h, test_h


def spec_features_k(tr_hvc, y_hvc, feats, k):
    """Top-k features by gain from one LGBM fit on HVC rows only."""
    dall = lgb.Dataset(tr_hvc[feats], y_hvc, categorical_feature=CAT_COLS)
    m = lgb.train(dict(KITCHEN_SINK, seed=42, num_threads=4), dall,
                  num_boost_round=800)
    gain = pd.Series(m.feature_importance(importance_type="gain"), index=feats)
    top = list(gain.sort_values(ascending=False).index[:k])
    print(f"specialist top-{k} features (gain): {top[:5]}...", flush=True)
    return top


def nested_split(groups, y, seed=SPLIT_SEED):
    """Split customers into SELECTION / VERIFICATION halves, grouped and
    stratified by each customer's positive count (0..4). Returns boolean
    row masks (sel, ver)."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({"g": groups, "y": y})
    per_cust = df.groupby("g")["y"].sum()
    sel_cust, ver_cust = set(), set()
    for _, idx in per_cust.groupby(per_cust).groups.items():
        idx = np.array(list(idx))
        rng.shuffle(idx)
        half = len(idx) // 2
        sel_cust.update(idx[:half])
        ver_cust.update(idx[half:])
    g = pd.Series(groups)
    sel = g.isin(sel_cust).values
    ver = ~sel
    print(f"nested split: selection customers={len(sel_cust)} "
          f"rows={sel.sum()}, verification customers={len(ver_cust)} "
          f"rows={ver.sum()}", flush=True)
    print(f"  selection pos rate={y[sel].mean():.4f}, "
          f"verification pos rate={y[ver].mean():.4f}", flush=True)
    return sel, ver


def blend_hvc(base, spec, mask, beta):
    """Return base with HVC rows blended toward spec at weight beta."""
    out = base.copy()
    out[mask] = np.clip((1 - beta) * base[mask] + beta * spec[mask],
                        EPS, 1 - EPS)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="spec-more,spec-cb,blend,topk",
                    help="comma-separated subset: "
                         "spec-more,spec-cb,blend,topk")
    args = ap.parse_args()
    stages = set(args.stages.split(","))
    os.makedirs(OUT, exist_ok=True)
    st = load_state()

    st19 = json.load(open(os.path.join(OUT, "iter19_state.json")))
    spec_feats = st19["spec_feats"]
    assert len(spec_feats) == SPEC_TOPK, \
        f"spec_feats != {SPEC_TOPK}: {len(spec_feats)}"
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

    # ---- stage 1: more LGBM specialist seeds (4 -> 10) ----
    spec10_oof_h = spec10_test_h = None
    if "spec-more" in stages:
        oofs, tests = [], []
        for s in SPEC_SEEDS10:
            d = load_spec_npz(s)
            if d is not None:
                print(f"seed {s}: specialist LGBM loaded cached",
                      flush=True)
                oofs.append(d["oof"])
                tests.append(d["test"])
            else:
                oof_s, test_s = train_spec_seed_lgbm(
                    s, tr0, te0, y, feats_final, groups, hvc_tr, hvc_te,
                    spec_feats)
                oofs.append(oof_s)
                tests.append(test_s)
            st["spec10_seeds_done"] = sorted(
                st.get("spec10_seeds_done", []) + [s])
            save_state(st)
        spec10_oof_h = np.clip(np.mean(oofs, axis=0), EPS, 1 - EPS)
        spec10_test_h = np.clip(np.mean(tests, axis=0), EPS, 1 - EPS)
        print(f"\nSpecialist LGBM 10-seed mean HVC OOF "
              f"comb={score_oof(y[hvc_tr], spec10_oof_h):.5f} "
              f"(iter19 4-seed: "
              f"{st19['spec_oof_comb_hvc']:.5f})", flush=True)
        st["spec10_oof_comb_hvc"] = score_oof(y[hvc_tr], spec10_oof_h)
        save_state(st)
    else:
        oofs = [load_spec_npz(s) for s in SPEC_SEEDS10]
        assert all(d is not None for d in oofs), "missing spec10 npz"
        spec10_oof_h = np.clip(np.mean([d["oof"] for d in oofs], axis=0),
                               EPS, 1 - EPS)
        spec10_test_h = np.clip(np.mean([d["test"] for d in oofs], axis=0),
                                EPS, 1 - EPS)

    # 4-seed LGBM specialist (iter19 recipe) for the verification baseline
    oofs4 = [load_spec_npz(s) for s in SPEC_SEEDS4]
    spec4_oof_h = np.clip(np.mean([d["oof"] for d in oofs4], axis=0),
                          EPS, 1 - EPS)
    spec4_test_h = np.clip(np.mean([d["test"] for d in oofs4], axis=0),
                           EPS, 1 - EPS)

    # ---- stage 2: CatBoost specialist ----
    spec_cb_oof_h = spec_cb_test_h = None
    if "spec-cb" in stages:
        oofs, tests = [], []
        for s in CB_SEEDS:
            p = os.path.join(OUT, f"iter20_spec_cb_s{s}.npz")
            if os.path.exists(p):
                d = np.load(p)
                print(f"seed {s}: specialist CatBoost loaded cached",
                      flush=True)
                oofs.append(d["oof"])
                tests.append(d["test"])
            else:
                oof_s, test_s = train_spec_seed_cb(
                    s, tr0, te0, y, feats_final, groups, hvc_tr, hvc_te,
                    spec_feats)
                oofs.append(oof_s)
                tests.append(test_s)
            st["spec_cb_seeds_done"] = sorted(
                st.get("spec_cb_seeds_done", []) + [s])
            save_state(st)
        spec_cb_oof_h = np.clip(np.mean(oofs, axis=0), EPS, 1 - EPS)
        spec_cb_test_h = np.clip(np.mean(tests, axis=0), EPS, 1 - EPS)
        print(f"\nSpecialist CB {len(CB_SEEDS)}-seed mean HVC OOF "
              f"comb={score_oof(y[hvc_tr], spec_cb_oof_h):.5f}", flush=True)
        st["spec_cb_oof_comb_hvc"] = score_oof(y[hvc_tr], spec_cb_oof_h)
        save_state(st)
    else:
        oofs = [np.load(os.path.join(OUT, f"iter20_spec_cb_s{s}.npz"))
                for s in CB_SEEDS]
        spec_cb_oof_h = np.clip(np.mean([d["oof"] for d in oofs], axis=0),
                                EPS, 1 - EPS)
        spec_cb_test_h = np.clip(np.mean([d["test"] for d in oofs], axis=0),
                                 EPS, 1 - EPS)

    corr_cb_lgbm = float(np.corrcoef(spec_cb_oof_h, spec10_oof_h)[0, 1])
    print(f"corr(CB spec OOF, LGBM-10 spec OOF) on HVC rows = "
          f"{corr_cb_lgbm:.4f} "
          f"({'DIVERSITY OK' if corr_cb_lgbm < 0.99 else 'NO DIVERSITY'})",
          flush=True)
    st["corr_cb_lgbm_spec"] = corr_cb_lgbm
    save_state(st)

    # ---- stage 3: nested blend selection + verification + gate ----
    winner = None  # (recipe dict, full comb, ver comb)
    if "blend" in stages:
        oof18, test18, _, _ = reconstruct_base18(y)
        # iter19 recipe OOF (for the verification baseline): base18 with
        # HVC rows at beta=0.90 toward the 4-seed LGBM specialist.
        # Build full-length specialist OOF vectors (base value off-HVC).
        spec4_full = oof18.copy()
        spec4_full[hvc_tr] = spec4_oof_h
        spec10_full = oof18.copy()
        spec10_full[hvc_tr] = spec10_oof_h
        spec_cb_full = oof18.copy()
        spec_cb_full[hvc_tr] = spec_cb_oof_h
        oof19 = blend_hvc(oof18, spec4_full, hvc_tr, 0.90)
        print(f"\niter19 recipe recomputed full OOF comb="
              f"{score_oof(y, oof19):.5f} (reported {BAR19:.5f})",
              flush=True)

        sel, ver = nested_split(groups, y)
        st["nested_split_seed"] = SPLIT_SEED
        save_state(st)

        def comb_on(mask, p):
            return score_oof(y[mask], p[mask])

        # Candidate family 1: LGBM-10 specialist only, beta sweep.
        # Candidate family 2: two-specialist mix, w x beta sweep.
        cands = []  # (label, spec_full_oof, beta)
        for beta in BETA_GRID:
            cands.append((f"lgbm10 beta={beta:.2f}", spec10_full, beta))
        for w in MIX_GRID:
            if w in (0.0, 1.0):
                continue  # duplicates of family 1 / CB-only (added below)
            spec_mix = w * spec10_full + (1 - w) * spec_cb_full
            for beta in [0.85, 0.90, 0.95]:
                cands.append((f"mix w_lgbm={w:.2f} beta={beta:.2f}",
                              spec_mix, beta))
        for beta in BETA_GRID:
            cands.append((f"cb-only beta={beta:.2f}", spec_cb_full, beta))

        print("\n--- nested SELECTION half: sweep (pick best here) ---",
              flush=True)
        sel_results = []
        for label, spec_full, beta in cands:
            p = blend_hvc(oof18, spec_full, hvc_tr, beta)
            sc_sel = comb_on(sel, p)
            sel_results.append((sc_sel, label, spec_full, beta))
            print(f"  {label}: selection comb={sc_sel:.5f}", flush=True)
        sel_results.sort(key=lambda t: -t[0])
        best_sel_sc, best_label, best_spec, best_beta = sel_results[0]
        print(f"SELECTION pick: {best_label} comb={best_sel_sc:.5f}",
              flush=True)
        st["selection_pick"] = {"label": best_label, "beta": best_beta,
                                "sel_comb": best_sel_sc}
        save_state(st)

        # Verification half: picked recipe vs iter19 recipe, same rows.
        p_pick = blend_hvc(oof18, best_spec, hvc_tr, best_beta)
        ver_pick = comb_on(ver, p_pick)
        ver_19 = comb_on(ver, oof19)
        sel_19 = comb_on(sel, oof19)
        full_pick = score_oof(y, p_pick)
        full_19 = score_oof(y, oof19)
        print("\n--- nested VERIFICATION half: picked vs iter19 recipe ---",
              flush=True)
        print(f"  iter19 recipe: selection={sel_19:.5f} "
              f"verification={ver_19:.5f} full={full_19:.5f}", flush=True)
        print(f"  iter20 picked ({best_label}): selection={best_sel_sc:.5f} "
              f"verification={ver_pick:.5f} full={full_pick:.5f}",
              flush=True)
        print(f"  verification delta (picked - iter19) = "
              f"{ver_pick - ver_19:+.5f}", flush=True)
        print(f"  full-OOF delta vs bar {BAR19:.5f} = "
              f"{full_pick - BAR19:+.5f}", flush=True)
        st["verification"] = {"picked_label": best_label, "beta": best_beta,
                              "ver_pick": ver_pick, "ver_19": ver_19,
                              "full_pick": full_pick, "full_19": full_19}
        save_state(st)

        gate_ver = ver_pick > ver_19
        gate_full = full_pick > BAR19
        print(f"\nGATE: verification beat iter19? {gate_ver}; "
              f"full OOF beat {BAR19}? {gate_full}", flush=True)
        if gate_ver and gate_full:
            # test predictions with the picked recipe
            spec4_test_full = test18.copy()
            spec10_test_full = test18.copy()
            spec_cb_test_full = test18.copy()
            spec10_test_full[hvc_te] = spec10_test_h
            spec_cb_test_full[hvc_te] = spec_cb_test_h
            if best_label.startswith("lgbm10"):
                spec_test = spec10_test_full
            elif best_label.startswith("cb-only"):
                spec_test = spec_cb_test_full
            else:  # mix
                w = float(best_label.split("w_lgbm=")[1].split()[0])
                spec_test = (w * spec10_test_full
                             + (1 - w) * spec_cb_test_full)
            test_f = test18.copy()
            test_f[hvc_te] = np.clip(
                (1 - best_beta) * test18[hvc_te]
                + best_beta * spec_test[hvc_te], 1e-6, 1 - 1e-6)
            assert len(test_f) == 30000 and not np.isnan(test_f).any()
            path = os.path.join(OUT, "iter20.csv")
            pd.DataFrame({ID_COL: te_ids, "Target": test_f}).to_csv(
                path, index=False)
            print(f"Saved submissions/iter20.csv (test mean "
                  f"{test_f.mean():.4f}, full honest OOF comb "
                  f"{full_pick:.5f})", flush=True)
            winner = {"label": best_label, "beta": best_beta,
                      "full_comb": full_pick, "ver_comb": ver_pick,
                      "path": path}
            st["winner"] = winner
            save_state(st)
        else:
            print("\nNo winner under the nested gate: no file written.",
                  flush=True)

    # ---- stage 4: top-k feature-count tune (nested protocol) ----
    if "topk" in stages and winner is not None:
        print("\n--- stage topk: only runs if blend stage produced a "
              "winner; checking whether a different specialist feature "
              "count beats it under the nested gate ---", flush=True)
        oof18, test18, _, _ = reconstruct_base18(y)
        spec4_full = oof18.copy()
        spec4_full[hvc_tr] = spec4_oof_h
        oof19 = blend_hvc(oof18, spec4_full, hvc_tr, 0.90)
        sel, ver = nested_split(groups, y)

        def comb_on(mask, p):
            return score_oof(y[mask], p[mask])

        tr42, te42 = add_te_grouped(tr0.copy(), te0.copy(), y, groups,
                                    seed=42)
        best_topk = None
        for k in TOPK_VALUES:
            feats_k = spec_features_k(tr42.loc[hvc_tr], y[hvc_tr],
                                      feats_final, k)
            oofs, tests = [], []
            for s in SPEC_SEEDS4:
                oof_s, test_s = train_spec_seed_lgbm(
                    s, tr0, te0, y, feats_final, groups, hvc_tr, hvc_te,
                    feats_k, topk=k)
                oofs.append(oof_s)
                tests.append(test_s)
            oof_k_h = np.clip(np.mean(oofs, axis=0), EPS, 1 - EPS)
            test_k_h = np.clip(np.mean(tests, axis=0), EPS, 1 - EPS)
            spec_k_full = oof18.copy()
            spec_k_full[hvc_tr] = oof_k_h
            print(f"top-{k} specialist (4-seed) HVC OOF comb="
                  f"{score_oof(y[hvc_tr], oof_k_h):.5f}", flush=True)
            for beta in BETA_GRID:
                p = blend_hvc(oof18, spec_k_full, hvc_tr, beta)
                sc_sel = comb_on(sel, p)
                print(f"  top-{k} beta={beta:.2f}: selection "
                      f"comb={sc_sel:.5f}", flush=True)
                if best_topk is None or sc_sel > best_topk[0]:
                    best_topk = (sc_sel, k, beta, spec_k_full, test_k_h)
        sc_sel, k_best, beta_best, spec_k_full, test_k_h = best_topk
        p_k = blend_hvc(oof18, spec_k_full, hvc_tr, beta_best)
        ver_k = comb_on(ver, p_k)
        ver_19 = comb_on(ver, oof19)
        full_k = score_oof(y, p_k)
        print(f"\ntopk pick: top-{k_best} beta={beta_best:.2f} "
              f"selection={sc_sel:.5f} verification={ver_k:.5f} "
              f"(iter19 {ver_19:.5f}) full={full_k:.5f}", flush=True)
        if ver_k > ver_19 and full_k > BAR19 \
                and full_k > winner["full_comb"]:
            spec_k_test_full = test18.copy()
            spec_k_test_full[hvc_te] = test_k_h
            test_f = test18.copy()
            test_f[hvc_te] = np.clip(
                (1 - beta_best) * test18[hvc_te]
                + beta_best * spec_k_test_full[hvc_te], 1e-6, 1 - 1e-6)
            assert len(test_f) == 30000 and not np.isnan(test_f).any()
            path = os.path.join(OUT, "iter20.csv")
            pd.DataFrame({ID_COL: te_ids, "Target": test_f}).to_csv(
                path, index=False)
            print(f"topk OVERWRITES iter20.csv with top-{k_best} "
                  f"beta={beta_best:.2f} (test mean {test_f.mean():.4f}, "
                  f"full honest OOF comb {full_k:.5f})", flush=True)
            winner = {"label": f"lgbm top-{k_best} beta={beta_best:.2f}",
                      "beta": beta_best, "full_comb": full_k,
                      "ver_comb": ver_k, "path": path}
            st["winner"] = winner
            save_state(st)
        else:
            print("topk does not beat the blend-stage winner under the "
                  "nested gate; iter20.csv unchanged.", flush=True)
        st["topk"] = {"k": k_best, "beta": beta_best,
                      "full_comb": full_k, "ver_comb": ver_k}
        save_state(st)

    # ---- results file ----
    with open(os.path.join(OUT, "iter20_results.txt"), "w") as f:
        f.write("ITERATION 20 results (HONEST customer-grouped CV, "
                "NESTED selection/verification protocol)\n")
        f.write(f"iter19 bar (full OOF): {BAR19:.5f}; iter19 public "
                f"0.72209 BELOW iter18 0.72354 -- OOF gain did not "
                f"transfer; beta was selection-biased.\n")
        f.write(f"corr(CB spec, LGBM-10 spec) HVC OOF = "
                f"{corr_cb_lgbm:.4f}\n")
        f.write(f"state: {json.dumps(st, default=str)}\n")
        if winner:
            f.write(f"WINNER: {winner['label']} beta={winner['beta']:.2f} "
                    f"full honest OOF comb={winner['full_comb']:.5f} "
                    f"(bar {BAR19:.5f}) verification comb="
                    f"{winner['ver_comb']:.5f} path={winner['path']}\n")
        else:
            f.write("NO WINNER under the nested gate; no file written.\n")
    print("EXIT:0")


if __name__ == "__main__":
    main()
