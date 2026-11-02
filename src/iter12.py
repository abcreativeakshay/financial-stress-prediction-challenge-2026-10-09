"""Iteration 12: hyperparam retune -> greedy FE pass 2 -> seed ensemble + blend re-tune.

Bar: iter11 OOF combined 0.81514 (6-seed ensemble of iter8 hybrid blend,
LGBM+CatBoost, 279 feats). Current best public 0.70415; leader 0.74293.

Stage A (hyperparam retune): kitchen_sink was tuned on the old 216-feature
set. Focused random search (~12 configs) on the 279-feature set, LGBM-only,
seed 42, vs iter8 LGBM seed-42 reference (0.81080, cached).
Stage B (greedy FE pass 2): ~100 new candidates (deeper liquidity ratios,
cross-month interactions, trajectory shapes, flow volatility, unsupervised
groupby stats), gain-ranked batches of 20 on CV, keep winners.
Stage C (seeds + blend): best single config -> 6-seed LGBM; blend with
cached CatBoost seeds (retrained on new feats only if Stage C-step1 misses);
re-tune hybrid geo/arith weights + recipe A/B. Submit file only if the
final ensemble beats 0.81514.

RESUME-SAFE: Stage A logs to submissions/iter12_search.csv; Stage B state in
submissions/iter12_state.json; Stage C per-seed npz skipped when present.

Rules: NEVER submits (parent handles browser). Commits left to the runner.
"""
import os
import sys
import json
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iter6
import iter8
from iter6 import (add_features_base, add_features_expanded, add_target_encoding,
                   run_lgbm, KITCHEN_SINK, CAT_COLS, N_SPLITS, TARGET, ID_COL,
                   DATA, OUT, EPS, combined)
from iter8 import add_liquidity_features, score_oof, run_catboost

BAR_FINAL = 0.81514   # iter11 6-seed
BAR_ITER9 = 0.81472   # iter9 3-seed
SEEDS6 = [42, 7, 2025, 5, 11, 777]
SEEDS3 = [42, 7, 2025]
MAX_FEATURES = 340


def build_279(seed=42):
    """The iter8 279-feature set (base + iter6 kept 50 + 13 liquidity). TE folds at `seed`."""
    iter6.SEED = seed
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
    assert not [c for c in feats if c not in tr.columns], "missing feats"
    y = tr[TARGET].values
    tr, te, _ = add_target_encoding(tr, te, y)
    print(f"279-feat set built (seed {seed}): {len(feats)} feats", flush=True)
    return tr, te, y, feats, bal_p25


def add_features_v2(tr, te, bal_p25):
    """Second feature battery (~100 candidates). Unsupervised; groupby maps fit on train only."""
    new_tr, new_te = {}, {}

    def add(name, v_tr, v_te):
        if name in tr.columns or name in new_tr:
            raise RuntimeError(f"name collision: {name}")
        new_tr[name] = np.asarray(v_tr, dtype=float)
        new_te[name] = np.asarray(v_te, dtype=float)

    def monthly(df, metric):
        return df[[f"m{i}_{metric}" for i in range(1, 7)]].values.astype(float)

    b_tr, b_te = monthly(tr, "daily_avg_bal"), monthly(te, "daily_avg_bal")
    wd_tr, wd_te = monthly(tr, "withdraw_total_value"), monthly(te, "withdraw_total_value")
    dp_tr, dp_te = monthly(tr, "deposit_total_value"), monthly(te, "deposit_total_value")
    rc_tr, rc_te = monthly(tr, "received_total_value"), monthly(te, "received_total_value")
    ms_tr, ms_te = monthly(tr, "mm_send_total_value"), monthly(te, "mm_send_total_value")
    pb_tr, pb_te = monthly(tr, "paybill_total_value"), monthly(te, "paybill_total_value")
    mp_tr, mp_te = monthly(tr, "merchantpay_total_value"), monthly(te, "merchantpay_total_value")
    tf_tr, tf_te = monthly(tr, "transfer_from_bank_total_value"), monthly(te, "transfer_from_bank_total_value")

    def pack(*arrs):
        # arrs: (a_tr, a_te) tuples -> stack newest-first per df
        return (np.stack([a[0] for a in arrs], axis=1),
                np.stack([a[1] for a in arrs], axis=1))

    inflow_tr = dp_tr + rc_tr
    inflow_te = dp_te + rc_te
    outflow_tr = wd_tr + ms_tr + pb_tr + mp_tr
    outflow_te = wd_te + ms_te + pb_te + mp_te
    outflow2_tr = outflow_tr + tf_tr
    outflow2_te = outflow_te + tf_te
    net_tr = inflow_tr - outflow_tr
    net_te = inflow_te - outflow_te
    balmean_tr = np.abs(b_tr.mean(axis=1))
    balmean_te = np.abs(b_te.mean(axis=1))

    # 1. per-month withdraw pressure m2..m6 (m1 exists)
    for i in range(1, 6):
        add(f"v2_wd_pressure_m{i + 1}",
            wd_tr[:, i] / np.maximum(b_tr[:, i], 1.0),
            wd_te[:, i] / np.maximum(b_te[:, i], 1.0))
    # 2. aggregate ratios
    add("v2_outflow6_div_balmean", outflow_tr.sum(1) / np.maximum(balmean_tr, 1.0),
        outflow_te.sum(1) / np.maximum(balmean_te, 1.0))
    add("v2_inflow6_div_balmean", inflow_tr.sum(1) / np.maximum(balmean_tr, 1.0),
        inflow_te.sum(1) / np.maximum(balmean_te, 1.0))
    add("v2_netflow6_div_balmean", net_tr.sum(1) / np.maximum(balmean_tr, 1.0),
        net_te.sum(1) / np.maximum(balmean_te, 1.0))
    add("v2_cumwd_div_cumdep", wd_tr.sum(1) / np.maximum(dp_tr.sum(1), 1.0),
        wd_te.sum(1) / np.maximum(dp_te.sum(1), 1.0))
    add("v2_max_outflow_div_mean", outflow_tr.max(1) / np.maximum(outflow_tr.mean(1), 1.0),
        outflow_te.max(1) / np.maximum(outflow_te.mean(1), 1.0))
    add("v2_outflow_cv", outflow_tr.std(1) / np.maximum(outflow_tr.mean(1), 1.0),
        outflow_te.std(1) / np.maximum(outflow_te.mean(1), 1.0))
    for i, m in enumerate([1, 2, 3]):
        add(f"v2_netflow_div_bal_m{m}", net_tr[:, i] / np.maximum(b_tr[:, i], 1.0),
            net_te[:, i] / np.maximum(b_te[:, i], 1.0))
    add("v2_wd_m1_concentration", wd_tr[:, 0] / np.maximum(wd_tr.sum(1), 1.0),
        wd_te[:, 0] / np.maximum(wd_te.sum(1), 1.0))
    add("v2_dep_m1_concentration", dp_tr[:, 0] / np.maximum(dp_tr.sum(1), 1.0),
        dp_te[:, 0] / np.maximum(dp_te.sum(1), 1.0))
    # 3. cross-month interactions
    add("v2_bal_drop_rate", (b_tr[:, 1] - b_tr[:, 0]) / np.maximum(b_tr[:, 0], 1.0),
        (b_te[:, 1] - b_te[:, 0]) / np.maximum(b_te[:, 0], 1.0))
    add("v2_outflow_surge", outflow_tr[:, 0] / np.maximum(outflow_tr[:, 1:].mean(1), 1.0),
        outflow_te[:, 0] / np.maximum(outflow_te[:, 1:].mean(1), 1.0))
    add("v2_outflow_accel", (outflow_tr[:, 0] - outflow_tr[:, 1]) - (outflow_tr[:, 1] - outflow_tr[:, 2]),
        (outflow_te[:, 0] - outflow_te[:, 1]) - (outflow_te[:, 1] - outflow_te[:, 2]))
    add("v2_inflow_accel", (inflow_tr[:, 0] - inflow_tr[:, 1]) - (inflow_tr[:, 1] - inflow_tr[:, 2]),
        (inflow_te[:, 0] - inflow_te[:, 1]) - (inflow_te[:, 1] - inflow_te[:, 2]))
    add("v2_wd_accel", (wd_tr[:, 0] - wd_tr[:, 1]) - (wd_tr[:, 1] - wd_tr[:, 2]),
        (wd_te[:, 0] - wd_te[:, 1]) - (wd_te[:, 1] - wd_te[:, 2]))
    add("v2_dep_accel", (dp_tr[:, 0] - dp_tr[:, 1]) - (dp_tr[:, 1] - dp_tr[:, 2]),
        (dp_te[:, 0] - dp_te[:, 1]) - (dp_te[:, 1] - dp_te[:, 2]))
    add("v2_wd_m1_x_m2", wd_tr[:, 0] * wd_tr[:, 1] / np.maximum(b_tr[:, 0], 1.0) ** 2,
        wd_te[:, 0] * wd_te[:, 1] / np.maximum(b_te[:, 0], 1.0) ** 2)
    add("v2_bal_m1m2_x_wdpress", (b_tr[:, 0] - b_tr[:, 1]) * (wd_tr[:, 0] / np.maximum(b_tr[:, 0], 1.0)),
        (b_te[:, 0] - b_te[:, 1]) * (wd_te[:, 0] / np.maximum(b_te[:, 0], 1.0)))
    add("v2_momentum_3m_bal", (b_tr[:, 0] - b_tr[:, 3]) / np.maximum(b_tr[:, 3], 1.0),
        (b_te[:, 0] - b_te[:, 3]) / np.maximum(b_te[:, 3], 1.0))
    add("v2_momentum_3m_flow", net_tr[:, 0] - net_tr[:, 3], net_te[:, 0] - net_te[:, 3])
    add("v2_wd_spike", wd_tr[:, 0] / np.maximum(wd_tr[:, 1], 1.0),
        wd_te[:, 0] / np.maximum(wd_te[:, 1], 1.0))
    add("v2_dep_spike", dp_tr[:, 0] / np.maximum(dp_tr[:, 1], 1.0),
        dp_te[:, 0] / np.maximum(dp_te[:, 1], 1.0))
    add("v2_pb_spike", pb_tr[:, 0] / np.maximum(pb_tr[:, 1:].mean(1), 1.0),
        pb_te[:, 0] / np.maximum(pb_te[:, 1:].mean(1), 1.0))
    # 4. trajectory shapes
    add("v2_conv_recent", b_tr[:, 0] - 2 * b_tr[:, 1] + b_tr[:, 2],
        b_te[:, 0] - 2 * b_te[:, 1] + b_te[:, 2])
    add("v2_conv_old", b_tr[:, 3] - 2 * b_tr[:, 4] + b_tr[:, 5],
        b_te[:, 3] - 2 * b_te[:, 4] + b_te[:, 5])
    add("v2_drawdown_dur", np.argmax(b_tr, axis=1).astype(float),
        np.argmax(b_te, axis=1).astype(float))
    add("v2_recovery", (b_tr[:, 0] - b_tr.min(1)) / np.maximum(b_tr.max(1) - b_tr.min(1), 1.0),
        (b_te[:, 0] - b_te.min(1)) / np.maximum(b_te.max(1) - b_te.min(1), 1.0))
    add("v2_second_min", np.partition(b_tr, 1, axis=1)[:, 1],
        np.partition(b_te, 1, axis=1)[:, 1])
    add("v2_second_max", np.partition(b_tr, 4, axis=1)[:, 4],
        np.partition(b_te, 4, axis=1)[:, 4])
    add("v2_n_below_mean", (b_tr < b_tr.mean(1, keepdims=True)).sum(1).astype(float),
        (b_te < b_te.mean(1, keepdims=True)).sum(1).astype(float))
    add("v2_range_div_mean", (b_tr.max(1) - b_tr.min(1)) / np.maximum(balmean_tr, 1.0),
        (b_te.max(1) - b_te.min(1)) / np.maximum(balmean_te, 1.0))
    add("v2_bal_cv", b_tr.std(1) / np.maximum(balmean_tr, 1.0),
        b_te.std(1) / np.maximum(balmean_te, 1.0))
    # 5. flow volatility
    add("v2_netflow_std_div_bal", net_tr.std(1) / np.maximum(balmean_tr, 1.0),
        net_te.std(1) / np.maximum(balmean_te, 1.0))
    add("v2_inflow_std_div_bal", inflow_tr.std(1) / np.maximum(balmean_tr, 1.0),
        inflow_te.std(1) / np.maximum(balmean_te, 1.0))
    mom_tr = np.diff(net_tr[:, ::-1], axis=1)  # m1-m2, m2-m3... newest-first diffs
    mom_te = np.diff(net_te[:, ::-1], axis=1)
    add("v2_max_netflow_drop", (-mom_tr).max(1), (-mom_te).max(1))
    add("v2_n_neg_netflow", (net_tr < 0).sum(1).astype(float),
        (net_te < 0).sum(1).astype(float))
    add("v2_flow_vol", (inflow_tr + outflow_tr).std(1) / np.maximum(balmean_tr, 1.0),
        (inflow_te + outflow_te).std(1) / np.maximum(balmean_te, 1.0))
    add("v2_n_zero_flow", (np.abs(net_tr) < 1).sum(1).astype(float),
        (np.abs(net_te) < 1).sum(1).astype(float))
    # max consecutive outflow>inflow months (vectorized over 6 steps)
    def streak(cond):
        best = cur = np.zeros(cond.shape[0])
        for j in range(cond.shape[1]):
            cur = np.where(cond[:, j], cur + 1, 0)
            best = np.maximum(best, cur)
        return best
    add("v2_outflow_streak", streak(outflow_tr > inflow_tr), streak(outflow_te > inflow_te))
    add("v2_dep_consistency", (dp_tr > 0).sum(1).astype(float),
        (dp_te > 0).sum(1).astype(float))
    add("v2_wd_consistency", (wd_tr > 0).sum(1).astype(float),
        (wd_te > 0).sum(1).astype(float))
    # 6. runway with broader outflow (incl. transfer_from_bank)
    run2_tr = b_tr / np.maximum(outflow2_tr, 1.0)
    run2_te = b_te / np.maximum(outflow2_te, 1.0)
    for i in range(6):
        add(f"v2_runway2_m{i + 1}", run2_tr[:, i], run2_te[:, i])
    add("v2_min_runway2", run2_tr.min(1), run2_te.min(1))
    add("v2_n_runway_lt1", (run2_tr < 1).sum(1).astype(float),
        (run2_te < 1).sum(1).astype(float))
    # 7. unsupervised groupby stats (train-fit maps)
    balmean_raw_tr = tr["bal_mean"].values
    balmean_raw_te = te["bal_mean"].values
    rw1_tr = tr["liq_runway_m1"].values
    rw1_te = te["liq_runway_m1"].values
    nf_tr = tr["netflow_sum"].values
    nf_te = te["netflow_sum"].values

    def grp_map(key_tr, key_te, val_tr, name):
        s_tr = pd.Series(val_tr).groupby(key_tr.astype(str).values).mean()
        fill = float(np.mean(val_tr))
        add(name,
            key_tr.astype(str).map(s_tr).fillna(fill).values,
            key_te.astype(str).map(s_tr).fillna(fill).values)

    for c in ["segment", "region", "earning_pattern", "gender", "smartphone"]:
        grp_map(tr[c], te[c], balmean_raw_tr, f"v2_grp_{c}_balmean")
    for c in ["segment", "region", "earning_pattern"]:
        grp_map(tr[c], te[c], rw1_tr, f"v2_grp_{c}_runway")
    grp_map(tr["segment"].astype(str) + "_" + tr["region"].astype(str),
            te["segment"].astype(str) + "_" + te["region"].astype(str),
            balmean_raw_tr, "v2_grp_seg_x_reg_balmean")
    grp_map(tr["segment"].astype(str) + "_" + tr["earning_pattern"].astype(str),
            te["segment"].astype(str) + "_" + te["earning_pattern"].astype(str),
            nf_tr, "v2_grp_seg_x_earn_netflow")
    # 8. more ratios
    arpu_tr, arpu_te = tr["arpu"].values, te["arpu"].values
    add("v2_arpu_div_balmean", arpu_tr / np.maximum(balmean_tr, 1.0),
        arpu_te / np.maximum(balmean_te, 1.0))
    t = np.arange(6)
    tc = t - t.mean()
    add("v2_age_x_netflow_slope",
        tr["age"].values * ((net_tr[:, ::-1] * tc).sum(1) / (tc ** 2).sum()),
        te["age"].values * ((net_te[:, ::-1] * tc).sum(1) / (tc ** 2).sum()))
    act_tr, act_te = tr["x_90_d_activity_rate"].values, te["x_90_d_activity_rate"].values
    add("v2_activity_x_wdpress", act_tr * (wd_tr[:, 0] / np.maximum(b_tr[:, 0], 1.0)),
        act_te * (wd_te[:, 0] / np.maximum(b_te[:, 0], 1.0)))
    add("v2_paybill_share", pb_tr.sum(1) / np.maximum(outflow_tr.sum(1), 1.0),
        pb_te.sum(1) / np.maximum(outflow_te.sum(1), 1.0))
    add("v2_mmsend_share", ms_tr.sum(1) / np.maximum(outflow_tr.sum(1), 1.0),
        ms_te.sum(1) / np.maximum(outflow_te.sum(1), 1.0))
    add("v2_tfb_div_inflow", tf_tr.sum(1) / np.maximum(inflow_tr.sum(1), 1.0),
        tf_te.sum(1) / np.maximum(inflow_te.sum(1), 1.0))
    add("v2_recv_div_dep", rc_tr.sum(1) / np.maximum(dp_tr.sum(1), 1.0),
        rc_te.sum(1) / np.maximum(dp_te.sum(1), 1.0))
    add("v2_bal_m1_div_inflow6", b_tr[:, 0] / np.maximum(inflow_tr.sum(1), 1.0),
        b_te[:, 0] / np.maximum(inflow_te.sum(1), 1.0))
    add("v2_inflow_diversity",
        1 - np.maximum(dp_tr.sum(1), rc_tr.sum(1)) / np.maximum(inflow_tr.sum(1), 1.0),
        1 - np.maximum(dp_te.sum(1), rc_te.sum(1)) / np.maximum(inflow_te.sum(1), 1.0))
    mxo_tr = np.maximum.reduce([wd_tr.sum(1), ms_tr.sum(1), pb_tr.sum(1), mp_tr.sum(1)])
    mxo_te = np.maximum.reduce([wd_te.sum(1), ms_te.sum(1), pb_te.sum(1), mp_te.sum(1)])
    add("v2_outflow_diversity", 1 - mxo_tr / np.maximum(outflow_tr.sum(1), 1.0),
        1 - mxo_te / np.maximum(outflow_te.sum(1), 1.0))
    add("v2_savings_rate", net_tr.sum(1) / np.maximum(inflow_tr.sum(1), 1.0),
        net_te.sum(1) / np.maximum(inflow_te.sum(1), 1.0))
    add("v2_m1_coverage", inflow_tr[:, 0] / np.maximum(outflow_tr[:, 0], 1.0),
        inflow_te[:, 0] / np.maximum(outflow_te[:, 0], 1.0))
    add("v2_inflow_coverage", inflow_tr.sum(1) / np.maximum(outflow_tr.sum(1), 1.0),
        inflow_te.sum(1) / np.maximum(outflow_te.sum(1), 1.0))
    add("v2_bal_m1_div_arpu12", b_tr[:, 0] / np.maximum(arpu_tr * 12, 1.0),
        b_te[:, 0] / np.maximum(arpu_te * 12, 1.0))
    add("v2_bal_below_p25", (b_tr[:, 0] < bal_p25).astype(float),
        (b_te[:, 0] < bal_p25).astype(float))
    add("v2_deep_stress",
        ((wd_tr[:, 0] / np.maximum(b_tr[:, 0], 1.0) > 1.0) & (rw1_tr < 1.0)).astype(float),
        ((wd_te[:, 0] / np.maximum(b_te[:, 0], 1.0) > 1.0) & (rw1_te < 1.0)).astype(float))
    add("v2_stress_proxy",
        (wd_tr[:, 0] / np.maximum(b_tr[:, 0], 1.0)) * (1 + (net_tr < 0).sum(1) / 6.0),
        (wd_te[:, 0] / np.maximum(b_te[:, 0], 1.0)) * (1 + (net_te < 0).sum(1) / 6.0))
    # 9. EWMA variants (alpha 0.3 / 0.9, oldest-first)
    def ewma(a, alpha):
        out = np.zeros_like(a)
        out[:, 0] = a[:, 0]
        for j in range(1, a.shape[1]):
            out[:, j] = alpha * a[:, j] + (1 - alpha) * out[:, j - 1]
        return out[:, -1]
    for alpha in (0.3, 0.9):
        add(f"v2_wd_ewma{int(alpha * 100):02d}", ewma(wd_tr[:, ::-1], alpha),
            ewma(wd_te[:, ::-1], alpha))
        add(f"v2_netflow_ewma{int(alpha * 100):02d}", ewma(net_tr[:, ::-1], alpha),
            ewma(net_te[:, ::-1], alpha))

    for k, v in new_tr.items():
        tr[k] = v
    for k, v in new_te.items():
        te[k] = v
    names = list(new_tr.keys())
    print(f"v2 features generated: {len(names)}", flush=True)
    return tr, te, names


# ---------------------------------------------------------------- Stage A
SEARCH_A_PATH = os.path.join(OUT, "iter12_search.csv")


def sample_search_configs(n=12):
    rng = random.Random(0)
    choices = {
        "num_leaves": [21, 31, 47, 63, 95],
        "min_child_samples": [20, 40, 80, 120],
        "feature_fraction": [0.6, 0.8, 1.0],
        "lambda_l1": [0.1, 0.5, 1.0, 2.0],
        "lambda_l2": [1.0, 3.0, 5.0, 8.0],
        "learning_rate": [0.02, 0.03, 0.05],
    }
    cfgs = []
    for i in range(n):
        cfgs.append((f"cfg{i:02d}", {k: rng.choice(v) for k, v in choices.items()}))
    return cfgs


def log_search(name, ll, auc, sc, mean_iter, overrides):
    pd.DataFrame([{"config": name, "ll": ll, "auc": auc, "combined": sc,
                   "mean_iter": mean_iter, "overrides": str(overrides)}]).to_csv(
        SEARCH_A_PATH, mode="a", header=not os.path.exists(SEARCH_A_PATH),
        index=False)


def load_search_done():
    if not os.path.exists(SEARCH_A_PATH):
        return {}
    df = pd.read_csv(SEARCH_A_PATH)
    return {r["config"]: (r["ll"], r["auc"], r["combined"], r["overrides"])
            for _, r in df.iterrows()}


def stage_a(tr, te, y, feats):
    """Hyperparam retune on 279 feats, LGBM-only, seed 42. Returns (best_params, best_score)."""
    print("\n=== Stage A: hyperparam retune (279 feats, LGBM-only, seed 42) ===", flush=True)
    iter6.SEED = 42
    X, Xte = tr[feats], te[feats]

    # reference: iter8 LGBM seed-42 OOF
    d = np.load(os.path.join(OUT, "iter8_lgbm.npz"))
    ref = score_oof(y, d["oof"])
    print(f"reference kitchen_sink (iter8 cache): combined={ref:.5f}", flush=True)

    done = load_search_done()
    results = {"kitchen_sink_ref": ref}
    best = {"name": "kitchen_sink_ref", "score": ref, "overrides": {}}

    for name, over in sample_search_configs():
        if name in done:
            ll, auc, sc, ov = done[name]
            print(f"{name} SKIPPED (done, comb={sc:.5f})", flush=True)
            cur_over = eval(ov) if isinstance(ov, str) else {}
        else:
            params = dict(KITCHEN_SINK)
            params.update(over)
            oof, test, mi = run_lgbm(X, y, Xte, params)
            ll = log_loss(y, oof)
            auc = roc_auc_score(y, oof)
            sc = combined(ll, auc)
            log_search(name, ll, auc, sc, mi, over)
            print(f"{name}: LL={ll:.5f} AUC={auc:.5f} comb={sc:.5f} "
                  f"mean_iter={mi:.0f} {over}", flush=True)
            cur_over = over
        if sc > best["score"]:
            best = {"name": name, "score": sc, "ll": ll, "auc": auc,
                    "overrides": cur_over}
        results[name] = sc

    # resolve overrides for the winner
    if best["name"] == "kitchen_sink_ref":
        best_params = dict(KITCHEN_SINK)
    else:
        best_params = dict(KITCHEN_SINK)
        best_params.update(best["overrides"])
    print(f"Stage A winner: {best['name']} combined={best['score']:.5f} "
          f"(ref {ref:.5f})", flush=True)
    return best_params, best["score"], results


# ---------------------------------------------------------------- Stage B
STATE_B_PATH = os.path.join(OUT, "iter12_state.json")
BATCH = 20
PATIENCE = 2


def save_state_b(state):
    with open(STATE_B_PATH, "w") as f:
        json.dump(state, f)


def load_state_b():
    if not os.path.exists(STATE_B_PATH):
        return None
    with open(STATE_B_PATH) as f:
        return json.load(f)


def stage_b(tr, te, y, feats279, best_params):
    """Greedy forward selection of v2 candidates in gain-ranked batches of 20.
    Returns (kept_v2, best_score). best_score starts at Stage-A winner's CV."""
    print("\n=== Stage B: greedy FE pass 2 ===", flush=True)
    iter6.SEED = 42
    state = load_state_b() or {}
    kept_v2 = state.get("kept_v2", [])
    done_batches = set(state.get("done_batches", []))
    # re-derive current best by evaluating base+kept (cheap: only if kept non-empty)
    X_base = tr[feats279]
    Xte_base = te[feats279]

    # build v2 features once
    tr2, te2, v2_names = add_features_v2(tr.copy(), te.copy(),
                                        float(np.percentile(
                                            tr[[f"m{i}_daily_avg_bal" for i in range(6, 0, -1)]].values.astype(float), 25)))
    # gain ranking from a full-data fit with best-A params
    if "ranked_v2" not in state:
        print("gain-ranking fit on full data (279+v2 feats)...", flush=True)
        Xg = tr2[feats279 + v2_names]
        dall = lgb.Dataset(Xg, y, categorical_feature=CAT_COLS)
        m = lgb.train(dict(best_params, num_threads=4), dall, num_boost_round=800)
        gain = pd.Series(m.feature_importance(importance_type="gain"),
                         index=feats279 + v2_names)
        ranked_v2 = [f for f in gain.sort_values(ascending=False).index
                     if f in v2_names]
        state["ranked_v2"] = ranked_v2
        save_state_b(state)
        print(f"ranked {len(ranked_v2)} v2 candidates by gain", flush=True)
    else:
        ranked_v2 = state["ranked_v2"]
        # ensure tr2/te2 have the features (rebuilt above); nothing else needed

    # current best: evaluate base+kept (or use Stage-A score if kept empty)
    if kept_v2:
        cand = feats279 + kept_v2
        oof, _, _ = run_lgbm(tr2[cand], y, te2[cand], best_params)
        best = score_oof(y, oof)
        print(f"resumed: base+{len(kept_v2)} kept -> combined={best:.5f}", flush=True)
    else:
        best = state.get("best", None)
        if best is None:
            oof, _, _ = run_lgbm(X_base, y, Xte_base, best_params)
            best = score_oof(y, oof)
            print(f"base (279, best-A params) combined={best:.5f}", flush=True)
    state["best"] = best

    batches = [ranked_v2[i:i + BATCH] for i in range(0, len(ranked_v2), BATCH)]
    no_gain = state.get("no_gain", 0)
    for bi, batch in enumerate(batches):
        if bi in done_batches:
            print(f"batch {bi}: skipped (done)", flush=True)
            continue
        if len(feats279) + len(kept_v2) + len(batch) > MAX_FEATURES:
            print(f"batch {bi}: skipped (would exceed {MAX_FEATURES})", flush=True)
            done_batches.add(bi)
            continue
        cand = feats279 + kept_v2 + batch
        oof, _, _ = run_lgbm(tr2[cand], y, te2[cand], best_params)
        sc = score_oof(y, oof)
        ll = log_loss(y, oof)
        auc = roc_auc_score(y, oof)
        improved = sc > best + 1e-6
        if improved:
            kept_v2.extend(batch)
            best = sc
            no_gain = 0
        else:
            no_gain += 1
        print(f"batch {bi} (+{len(batch)} -> {len(cand)} total): "
              f"LL={ll:.5f} AUC={auc:.5f} comb={sc:.5f} "
              f"{'KEPT' if improved else 'dropped'} (best {best:.5f})", flush=True)
        done_batches.add(bi)
        state.update({"kept_v2": kept_v2, "done_batches": sorted(done_batches),
                      "best": best, "no_gain": no_gain})
        save_state_b(state)
        if no_gain >= PATIENCE:
            print(f"early stop: {PATIENCE} non-improving batches", flush=True)
            break
    print(f"Stage B done: kept {len(kept_v2)} v2 features, best combined={best:.5f}",
          flush=True)
    return kept_v2, best


# ---------------------------------------------------------------- Stage C
def hybrid(a, b, w_geo, w_lgbm):
    g = np.sqrt(np.clip(a, EPS, 1) * np.clip(b, EPS, 1))
    arith = w_lgbm * a + (1 - w_lgbm) * b
    return w_geo * g + (1 - w_geo) * arith


def train_lgbm_seed(seed, tr0, te0, y, feats, params):
    """LGBM at `seed` (TE folds + CV folds seeded). Resume-safe npz."""
    iter6.SEED = seed
    p = os.path.join(OUT, f"iter12_lgbm_s{seed}.npz")
    if os.path.exists(p):
        d = np.load(p)
        print(f"seed {seed}: LGBM loaded cached", flush=True)
        return d["oof"], d["test"]
    tr = tr0.copy()
    te = te0.copy()
    tr, te, _ = add_target_encoding(tr, te, y)
    X, Xte = tr[feats], te[feats]
    prm = dict(params, seed=seed)
    print(f"seed {seed}: training LGBM...", flush=True)
    oof, test, mi = run_lgbm(X, y, Xte, prm)
    np.savez(p, oof=oof, test=test, iters=mi)
    print(f"seed {seed}: LGBM done, mean_iter={mi:.0f}, comb={score_oof(y, oof):.5f}",
          flush=True)
    return oof, test


def train_cb_seed(seed, tr0, te0, y, feats):
    """CatBoost at `seed` on the (possibly new) feature set. Resume-safe npz."""
    iter8.SEED = seed
    p = os.path.join(OUT, f"iter12_catboost_s{seed}.npz")
    if os.path.exists(p):
        d = np.load(p)
        print(f"seed {seed}: CatBoost loaded cached", flush=True)
        return d["oof"], d["test"]
    iter6.SEED = seed
    tr = tr0.copy()
    te = te0.copy()
    tr, te, _ = add_target_encoding(tr, te, y)
    X, Xte = tr[feats], te[feats]
    print(f"seed {seed}: training CatBoost (2000 iters)...", flush=True)
    oof, test, mi = run_catboost(X, y, Xte)
    np.savez(p, oof=oof, test=test, iters=mi)
    print(f"seed {seed}: CatBoost done, mean_iter={mi:.0f}, comb={score_oof(y, oof):.5f}",
          flush=True)
    return oof, test


def blend_bakeoff(y, oofs_l, tests_l, oofs_c, tests_c, seeds, tag):
    """Grid over w_geo x w_lgbm x recipe(A/B). Returns (best_spec, oof, test, score)."""
    cands = []
    for w_geo in (0.2, 0.3, 0.4, 0.5):
        for w_lgbm in (0.4, 0.5, 0.6):
            hyb_o = [hybrid(oofs_l[i], oofs_c[i], w_geo, w_lgbm) for i in range(len(seeds))]
            hyb_t = [hybrid(tests_l[i], tests_c[i], w_geo, w_lgbm) for i in range(len(seeds))]
            oof_a = np.clip(np.mean(hyb_o, axis=0), EPS, 1 - EPS)
            test_a = np.clip(np.mean(hyb_t, axis=0), EPS, 1 - EPS)
            sc_a = score_oof(y, oof_a)
            oof_b = np.clip(hybrid(np.mean(oofs_l, axis=0), np.mean(oofs_c, axis=0),
                                   w_geo, w_lgbm), EPS, 1 - EPS)
            test_b = np.clip(hybrid(np.mean(tests_l, axis=0), np.mean(tests_c, axis=0),
                                    w_geo, w_lgbm), EPS, 1 - EPS)
            sc_b = score_oof(y, oof_b)
            cands.append((f"A_wg{w_geo}_wl{w_lgbm}", sc_a, oof_a, test_a))
            cands.append((f"B_wg{w_geo}_wl{w_lgbm}", sc_b, oof_b, test_b))
    cands.sort(key=lambda r: -r[1])
    print(f"\nBlend bake-off [{tag}] top 5:", flush=True)
    for nm, sc, _, _ in cands[:5]:
        print(f"  {nm:22s} comb={sc:.5f}", flush=True)
    nm, sc, oof, test = cands[0]
    print(f"[{tag}] winner: {nm} comb={sc:.5f}", flush=True)
    return nm, oof, test, sc


def stage_c(tr0, te0, y, feats_final, best_params, feats_changed):
    """6-seed LGBM (+cached or retrained CatBoost), blend re-tune.
    Returns (oof_final, test_final, score, spec) or None if below bar."""
    print("\n=== Stage C: seed ensemble + blend re-tune ===", flush=True)

    oofs_l, tests_l, oofs_c, tests_c = [], [], [], []
    for s in SEEDS6:
        oof_l, test_l = train_lgbm_seed(s, tr0, te0, y, feats_final, best_params)
        oofs_l.append(oof_l)
        tests_l.append(test_l)
        if feats_changed:
            oof_c, test_c = train_cb_seed(s, tr0, te0, y, feats_final)
        else:
            d = np.load(os.path.join(OUT, f"iter9_catboost_s{s}.npz"))
            oof_c, test_c = d["oof"], d["test"]
            print(f"seed {s}: CatBoost reused iter9 cache (feats unchanged)", flush=True)
        oofs_c.append(oof_c)
        tests_c.append(test_c)

    nm, oof_f, test_f, sc = blend_bakeoff(y, oofs_l, tests_l, oofs_c, tests_c,
                                          SEEDS6, "6-seed")
    spec = {"blend": nm, "seeds": SEEDS6, "feats_changed": feats_changed,
            "lgbm_params": {k: v for k, v in best_params.items()
                            if k not in ("objective", "metric", "verbose")}}
    print(f"Stage C final: {nm} OOF comb={sc:.5f} "
          f"({'BEATS' if sc > BAR_FINAL else 'below'} bar {BAR_FINAL})", flush=True)
    if sc > BAR_FINAL:
        return oof_f, test_f, sc, spec
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    results_log = []

    # ---- build 279 set (seed 42) ----
    tr, te, y, feats279, bal_p25 = build_279(seed=42)

    # ---- Stage A ----
    best_params, bestA_score, search_res = stage_a(tr, te, y, feats279)
    results_log.append(f"Stage A winner combined={bestA_score:.5f} "
                       f"(ref 0.81080): { {k: v for k, v in best_params.items() if k in ('num_leaves','min_child_samples','feature_fraction','lambda_l1','lambda_l2','learning_rate')} }")

    # ---- Stage B ----
    kept_v2, bestB_score = stage_b(tr, te, y, feats279, best_params)
    results_log.append(f"Stage B kept {len(kept_v2)} v2 features, best combined={bestB_score:.5f}")
    if kept_v2:
        results_log.append("kept_v2: " + ", ".join(kept_v2))

    feats_final = feats279 + kept_v2
    feats_changed = len(kept_v2) > 0
    params_changed = any(best_params.get(k) != KITCHEN_SINK.get(k)
                         for k in ("num_leaves", "min_child_samples", "feature_fraction",
                                   "lambda_l1", "lambda_l2", "learning_rate"))
    print(f"\nfeats_changed={feats_changed} params_changed={params_changed}", flush=True)

    # ---- Stage C (skip if nothing changed: would reproduce iter11) ----
    final = None
    if feats_changed or params_changed:
        # tr0/te0 for Stage C: 279 base (TE re-done per seed inside trainers).
        # v2 features must be present in tr0/te0:
        if feats_changed:
            tr0, te0, _ = add_features_v2(tr.copy(), te.copy(), bal_p25)
        else:
            tr0, te0 = tr, te
        final = stage_c(tr0, te0, y, feats_final, best_params, feats_changed)
    else:
        print("Nothing changed vs iter11 inputs -> Stage C skipped (would reproduce 0.81514).",
              flush=True)

    with open(os.path.join(OUT, "iter12_results.txt"), "w") as f:
        f.write("ITERATION 12 results\n")
        f.write(f"bar (iter11 6-seed): {BAR_FINAL}\n")
        for line in results_log:
            f.write(line + "\n")
        f.write("\nStage A search (LGBM-only, seed 42, 279 feats):\n")
        for k in sorted(search_res, key=lambda k: -search_res[k]):
            f.write(f"  {k}: {search_res[k]:.5f}\n")

    if final is not None:
        oof_f, test_f, sc, spec = final
        test_f = np.clip(test_f, EPS, 1 - EPS)
        pd.DataFrame({ID_COL: te[ID_COL], "Target": test_f}).to_csv(
            os.path.join(OUT, "iter12.csv"), index=False)
        pd.DataFrame({"ID": tr[ID_COL], "oof_pred": oof_f, TARGET: y}).to_csv(
            os.path.join(OUT, "oof_iter12.csv"), index=False)
        with open(os.path.join(OUT, "iter12_results.txt"), "a") as f:
            f.write(f"\nFINAL: {spec['blend']} OOF comb={sc:.5f} BEATS {BAR_FINAL}\n")
            f.write(f"spec: {json.dumps(spec)}\n")
        print(f"Saved submissions/iter12.csv (test mean {test_f.mean():.4f}). "
              f"BEATS bar — ready to submit (NOT submitted here).", flush=True)
    else:
        with open(os.path.join(OUT, "iter12_results.txt"), "a") as f:
            f.write("\nFINAL: no candidate beat the bar; iter12.csv NOT written.\n")
        print("No candidate beat the bar; iter12.csv NOT written.", flush=True)
    print("EXIT:0")


if __name__ == "__main__":
    main()
