"""Iteration 14: v4 residual-targeted feature battery -> CatBoost depth-8 retest ->
pseudo-residual stacking -> 6->10 seeds (conditional) -> blend re-tune.

Bar: iter13 6-seed hybrid A_wg0.5_wl0.5 = 0.82116 (359 feats, public 0.71441).
Public: us 0.71441, friend 0.71882, leader 0.74293. 3 submission slots left today.

Residual analysis (src/resid14.py) found:
 - low-activity quintile worst (LL 0.2751); low-balance quintile worst (0.2915)
 - worst LL decile is 78.5% positive, mean p=0.33 -> model UNDER-predicts stress
 - FN (776) >> FP (227); FN have decent m1 balances (median 5559) -> hidden stress
 - HVC/LVC overrepresented in worst decile (+0.04 each); MVC underrepresented
 - arpu extremes (lowest & highest quintile) harder; young users slightly harder

RULES: 5-fold stratified CV seed 42, per-fold isotonic. NEVER submits
(parent handles browser). Resume-safe via submissions/iter14_state.json.
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_predict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iter6
import iter8
import iter12
import iter13
from iter6 import (add_target_encoding, run_lgbm, KITCHEN_SINK, CAT_COLS,
                   N_SPLITS, TARGET, ID_COL, DATA, OUT, EPS, combined)
from iter8 import score_oof, CAT_ORDER
from iter12 import hybrid, blend_bakeoff

BAR = 0.82116          # iter13 6-seed hybrid
SEEDS6 = [42, 7, 2025, 5, 11, 777]
SEEDS10 = SEEDS6 + [99, 1234, 2024, 31337]
BATCH = 20
PATIENCE = 2
MAX_FEATURES = 400
STATE14 = os.path.join(OUT, "iter14_state.json")
CB_REF = {"depth": 6, "learning_rate": 0.03, "iterations": 2000, "l2_leaf_reg": 3.0}


def save_state(st):
    with open(STATE14, "w") as f:
        json.dump(st, f)


def load_state():
    if os.path.exists(STATE14):
        with open(STATE14) as f:
            return json.load(f)
    return {}


def build_359(seed=42):
    """iter13's 359-feature set (319 + 40 kept v3), TE at `seed`."""
    tr319, te319, y, feats319, bal_p25 = iter13.build_319(seed=seed)
    with open(os.path.join(OUT, "iter13_state.json")) as f:
        kept_v3 = json.load(f)["stage2_kept"]
    tr2, te2, _ = iter13.add_features_v3(tr319.copy(), te319.copy())
    feats = feats319 + kept_v3
    missing = [c for c in feats if c not in tr2.columns]
    assert not missing, f"missing feats: {missing[:5]}"
    print(f"359-feat set built (seed {seed}): {len(feats)} feats", flush=True)
    return tr2, te2, y, feats, bal_p25


# ---------------------------------------------------------------- v4 battery
def add_features_v4(tr, te):
    """Fourth battery (~84 candidates, v4_ prefix), targeted at residual pockets.
    All unsupervised (no target use)."""
    new_tr, new_te = {}, {}

    def add(name, v_tr, v_te):
        if name in tr.columns or name in new_tr:
            raise RuntimeError(f"name collision: {name}")
        new_tr[name] = np.asarray(v_tr, dtype=float)
        new_te[name] = np.asarray(v_te, dtype=float)

    def monthly(df, metric):
        return df[[f"m{i}_{metric}" for i in range(1, 7)]].values.astype(float)

    b = monthly(tr, "daily_avg_bal")
    bte = monthly(te, "daily_avg_bal")
    t = np.arange(6, dtype=float)
    tc = t - t.mean()

    def slope(a):
        return (a * tc).sum(axis=1) / (tc ** 2).sum()

    cats = ["paybill", "merchantpay", "transfer_from_bank", "mm_send",
            "deposit", "withdraw", "received"]
    tv, tvte = {}, {}
    vol, volte = {}, {}
    for c in cats:
        tv[c] = monthly(tr, f"{c}_total_value")
        tvte[c] = monthly(te, f"{c}_total_value")
        vol[c] = monthly(tr, f"{c}_volume")
        volte[c] = monthly(te, f"{c}_volume")

    inflow = tv["deposit"] + tv["received"]
    inflowte = tvte["deposit"] + tvte["received"]
    outflow = tv["withdraw"] + tv["mm_send"] + tv["paybill"] + tv["merchantpay"]
    outflowte = (tvte["withdraw"] + tvte["mm_send"] + tvte["paybill"]
                 + tvte["merchantpay"])
    net = inflow - outflow
    nette = inflowte - outflowte
    balmean = np.abs(b.mean(axis=1))
    balmeante = np.abs(bte.mean(axis=1))
    act = tr["x_90_d_activity_rate"].values.astype(float)
    actte = te["x_90_d_activity_rate"].values.astype(float)
    age = tr["age"].values.astype(float)
    agete = te["age"].values.astype(float)
    arpu = tr["arpu"].values.astype(float)
    arpute = te["arpu"].values.astype(float)
    wd = tv["withdraw"]
    wdte = tvte["withdraw"]
    dp = tv["deposit"]
    dpte = tvte["deposit"]

    def cv(a):
        mu = np.abs(a.mean(axis=1))
        return a.std(axis=1) / np.maximum(mu, 1e-9)

    seg = tr["segment"].astype(str).values
    segte = te["segment"].astype(str).values
    seg_masks = {s: (seg == s).astype(float) for s in ("HVC", "LVC", "MVC")}
    seg_masks_te = {s: (segte == s).astype(float) for s in ("HVC", "LVC", "MVC")}

    # ---- A. activity-pocket interactions (10): low-activity users are worst
    low_act = (act < 0.3).astype(float)
    low_actte = (actte < 0.3).astype(float)
    runway_m1 = b[:, 0] / np.maximum(outflow[:, 0], 1.0)
    runway_m1te = bte[:, 0] / np.maximum(outflowte[:, 0], 1.0)
    net_slope = slope(net)
    net_slopete = slope(nette)
    signflips = (np.diff(np.sign(net), axis=1) != 0).sum(axis=1).astype(float)
    signflipste = (np.diff(np.sign(nette), axis=1) != 0).sum(axis=1).astype(float)
    add("v4_lowact_x_runway_m1", low_act * runway_m1, low_actte * runway_m1te)
    add("v4_lowact_x_netslope", low_act * net_slope, low_actte * net_slopete)
    add("v4_lowact_x_signflips", low_act * signflips, low_actte * signflipste)
    wd_mean = wd.mean(axis=1)
    wd_meante = wdte.mean(axis=1)
    add("v4_lowact_wd_spike", low_act * wd[:, 0] / np.maximum(wd_mean, 1.0),
        low_actte * wdte[:, 0] / np.maximum(wd_meante, 1.0))
    add("v4_act_x_dp_collapse",
        act * dp[:, 0] / np.maximum(dp.mean(axis=1), 1.0),
        actte * dpte[:, 0] / np.maximum(dpte.mean(axis=1), 1.0))
    bal_drop = (b.max(axis=1) - b[:, 0]) / np.maximum(b.max(axis=1), 1.0)
    bal_dropte = (bte.max(axis=1) - bte[:, 0]) / np.maximum(bte.max(axis=1), 1.0)
    add("v4_act_x_baldrop", act * bal_drop, actte * bal_dropte)
    add("v4_inact_x_outflow_cv", (1 - act) * cv(outflow), (1 - actte) * cv(outflowte))
    add("v4_act_quintile", pd.qcut(act, 5, labels=False, duplicates="drop").astype(float),
        pd.qcut(actte, 5, labels=False, duplicates="drop").astype(float))
    add("v4_lowact_x_wdaccel",
        low_act * ((wd[:, 0] - wd[:, 1]) - (wd[:, 1] - wd[:, 2])),
        low_actte * ((wdte[:, 0] - wdte[:, 1]) - (wdte[:, 1] - wdte[:, 2])))
    add("v4_lowact_x_savingsmin",
        low_act * ((inflow - outflow) / np.maximum(inflow, 1.0)).min(axis=1),
        low_actte * ((inflowte - outflowte) / np.maximum(inflowte, 1.0)).min(axis=1))

    # ---- B. segment-conditioned dynamics (12): HVC/LVC overrepresented in errors
    seg_feats = {
        "netslope": (net_slope, net_slopete),
        "baldrop": (bal_drop, bal_dropte),
        "wdpressure": (wd[:, 0] / np.maximum(b[:, 0], 1.0),
                       wdte[:, 0] / np.maximum(bte[:, 0], 1.0)),
        "runway": (runway_m1, runway_m1te),
    }
    for s in ("HVC", "LVC", "MVC"):
        for fname, (arr, arrte) in seg_feats.items():
            add(f"v4_seg{s}_x_{fname}", seg_masks[s] * arr, seg_masks_te[s] * arrte)

    # ---- C. hidden-stress / missed positives (12): decent balance, bad flows
    bal_med = np.median(b[:, 0])
    bal_medte = np.median(bte[:, 0])
    ok_bal = (b[:, 0] > bal_med).astype(float)
    ok_balte = (bte[:, 0] > bal_medte).astype(float)
    add("v4_okbal_x_negnetslope", ok_bal * np.minimum(net_slope, 0),
        ok_balte * np.minimum(net_slopete, 0))
    add("v4_outflow_growth_bal_flat",
        slope(outflow) - slope(b) / np.maximum(balmean, 1.0),
        slope(outflowte) - slope(bte) / np.maximum(balmeante, 1.0))
    dp_reg = 1.0 / (1.0 + cv(dp))
    dp_regte = 1.0 / (1.0 + cv(dpte))
    add("v4_dp_reg_collapse", dp_reg * dp[:, 0] / np.maximum(dp.mean(axis=1), 1.0),
        dp_regte * dpte[:, 0] / np.maximum(dpte.mean(axis=1), 1.0))
    recv_share = tv["received"] / np.maximum(inflow, 1.0)
    recv_sharete = tvte["received"] / np.maximum(inflowte, 1.0)
    add("v4_recvshare_shift", recv_share[:, 0] - recv_share.mean(axis=1),
        recv_sharete[:, 0] - recv_sharete.mean(axis=1))
    sr = (inflow - outflow) / np.maximum(inflow, 1.0)
    srte = (inflowte - outflowte) / np.maximum(inflowte, 1.0)
    add("v4_savings_deepneg", np.minimum(sr.min(axis=1), 0),
        np.minimum(srte.min(axis=1), 0))
    add("v4_wd_while_inflow_high",
        wd[:, 0] / np.maximum(inflow[:, 0], 1.0) * (inflow[:, 0] > np.median(inflow[:, 0])).astype(float),
        wdte[:, 0] / np.maximum(inflowte[:, 0], 1.0) * (inflowte[:, 0] > np.median(inflowte[:, 0])).astype(float))
    add("v4_netneg_x_balok", (net[:, 0] < 0).astype(float) * ok_bal,
        (nette[:, 0] < 0).astype(float) * ok_balte)
    add("v4_inflow_down_wd_up",
        np.maximum(0, -slope(inflow)) * np.maximum(0, slope(wd)),
        np.maximum(0, -slope(inflowte)) * np.maximum(0, slope(wdte)))
    add("v4_bal_flat_outflow_up",
        (np.abs(slope(b)) / np.maximum(balmean, 1.0) < 0.05).astype(float) * np.maximum(0, slope(outflow)),
        (np.abs(slope(bte)) / np.maximum(balmeante, 1.0) < 0.05).astype(float) * np.maximum(0, slope(outflowte)))
    add("v4_okbal_x_wdshare",
        ok_bal * wd[:, 0] / np.maximum(outflow[:, 0], 1.0),
        ok_balte * wdte[:, 0] / np.maximum(outflowte[:, 0], 1.0))
    add("v4_okbal_x_cvoutflow", ok_bal * cv(outflow), ok_balte * cv(outflowte))
    add("v4_dp_down_trend", np.minimum(slope(dp), 0), np.minimum(slope(dpte), 0))

    # ---- D. tail-risk indicators (12)
    add("v4_max_monthly_outflow", outflow.max(axis=1), outflowte.max(axis=1))
    add("v4_min_monthly_netflow", net.min(axis=1), nette.min(axis=1))
    add("v4_worst_netflow_div_bal", net.min(axis=1) / np.maximum(balmean, 1.0),
        nette.min(axis=1) / np.maximum(balmeante, 1.0))
    # consecutive negative-netflow months, newest first (vectorized streak:
    # count leading True months starting at m1)
    neg = (net < 0).astype(int)
    negte = (nette < 0).astype(int)
    s = np.zeros(len(tr)); st_ = np.zeros(len(te))
    for i in range(6):
        s = np.where((neg[:, i] == 1) & (s == i), i + 1, s)
        st_ = np.where((negte[:, i] == 1) & (st_ == i), i + 1, st_)
    add("v4_neg_netflow_streak", s, st_)
    # longest balance-decline run
    dcl = (np.diff(b[:, ::-1], axis=1) < 0).astype(int)  # oldest->newest diffs
    dclte = (np.diff(bte[:, ::-1], axis=1) < 0).astype(int)
    def longest_run(a):
        best = np.zeros(a.shape[0]); cur = np.zeros(a.shape[0])
        for i in range(a.shape[1]):
            cur = np.where(a[:, i] == 1, cur + 1, 0)
            best = np.maximum(best, cur)
        return best
    add("v4_bal_decline_run", longest_run(dcl), longest_run(dclte))
    # own-history z-scores of m1
    om, osd = outflow.mean(axis=1), outflow.std(axis=1)
    omte, osdte = outflowte.mean(axis=1), outflowte.std(axis=1)
    add("v4_outflow_z_m1", (outflow[:, 0] - om) / np.maximum(osd, 1.0),
        (outflowte[:, 0] - omte) / np.maximum(osdte, 1.0))
    bm, bsd = b.mean(axis=1), b.std(axis=1)
    bmte, bsdte = bte.mean(axis=1), bte.std(axis=1)
    add("v4_bal_z_m1", (b[:, 0] - bm) / np.maximum(bsd, 1.0),
        (bte[:, 0] - bmte) / np.maximum(bsdte, 1.0))
    nm_, nsd = net.mean(axis=1), net.std(axis=1)
    nmte, nsdte = nette.mean(axis=1), nette.std(axis=1)
    add("v4_netflow_z_m1", (net[:, 0] - nm_) / np.maximum(nsd, 1.0),
        (nette[:, 0] - nmte) / np.maximum(nsdte, 1.0))
    add("v4_maxdd_bal", (np.maximum.accumulate(b[:, ::-1], axis=1) - b[:, ::-1]).max(axis=1),
        (np.maximum.accumulate(bte[:, ::-1], axis=1) - bte[:, ::-1]).max(axis=1))
    add("v4_time_since_maxbal", np.argmax(b[:, ::-1], axis=1).astype(float),
        np.argmax(bte[:, ::-1], axis=1).astype(float))
    add("v4_bal_range_div_mean", (b.max(axis=1) - b.min(axis=1)) / np.maximum(balmean, 1.0),
        (bte.max(axis=1) - bte.min(axis=1)) / np.maximum(balmeante, 1.0))

    # ---- E. arpu-extreme interactions (8): both arpu tails are hard
    arpu_q = pd.qcut(arpu, 5, labels=False, duplicates="drop").astype(float)
    arpu_qte = pd.qcut(arpute, 5, labels=False, duplicates="drop").astype(float)
    arpu_tail = ((arpu_q == 0) | (arpu_q == 4)).astype(float)
    arpu_tailte = ((arpu_qte == 0) | (arpu_qte == 4)).astype(float)
    add("v4_arpu_tail_x_pressure", arpu_tail * wd[:, 0] / np.maximum(b[:, 0], 1.0),
        arpu_tailte * wdte[:, 0] / np.maximum(bte[:, 0], 1.0))
    add("v4_arpu_tail_x_netslope", arpu_tail * net_slope, arpu_tailte * net_slopete)
    add("v4_inflow_div_arpu", inflow.sum(1) / np.maximum(arpu, 1.0),
        inflowte.sum(1) / np.maximum(arpute, 1.0))
    add("v4_netflow_div_arpu", net.sum(1) / np.maximum(arpu, 1.0),
        nette.sum(1) / np.maximum(arpute, 1.0))
    add("v4_arpu_x_segment_LVC", arpu * seg_masks["LVC"], arpute * seg_masks_te["LVC"])
    add("v4_arpu_x_segment_HVC", arpu * seg_masks["HVC"], arpute * seg_masks_te["HVC"])
    add("v4_arpu_quintile", arpu_q, arpu_qte)
    add("v4_lowarpu_x_lowact", ((arpu_q == 0).astype(float)) * low_act,
        ((arpu_qte == 0).astype(float)) * low_actte)

    # ---- F. young-user dynamics (6)
    young = (age < 30).astype(float)
    youngte = (agete < 30).astype(float)
    add("v4_young_x_pressure", young * wd[:, 0] / np.maximum(b[:, 0], 1.0),
        youngte * wdte[:, 0] / np.maximum(bte[:, 0], 1.0))
    add("v4_young_x_netslope", young * net_slope, youngte * net_slopete)
    add("v4_young_x_wdaccel",
        young * ((wd[:, 0] - wd[:, 1]) - (wd[:, 1] - wd[:, 2])),
        youngte * ((wdte[:, 0] - wdte[:, 1]) - (wdte[:, 1] - wdte[:, 2])))
    add("v4_young_x_runway", young * runway_m1, youngte * runway_m1te)
    add("v4_age_x_pressure", age * wd[:, 0] / np.maximum(b[:, 0], 1.0),
        agete * wdte[:, 0] / np.maximum(bte[:, 0], 1.0))
    add("v4_age_quintile", pd.qcut(age, 5, labels=False, duplicates="drop").astype(float),
        pd.qcut(agete, 5, labels=False, duplicates="drop").astype(float))

    # ---- G. ticket / volume dynamics (8)
    for c, arr, arrte, varr, varrte in (("pb", tv["paybill"], tvte["paybill"], vol["paybill"], volte["paybill"]),
                                        ("wd", wd, wdte, vol["withdraw"], volte["withdraw"])):
        tick = arr / np.maximum(varr, 1.0)
        tickte = arrte / np.maximum(varrte, 1.0)
        add(f"v4_ticket_{c}_accel",
            (tick[:, 0] - tick[:, 1]) - (tick[:, 1] - tick[:, 2]),
            (tickte[:, 0] - tickte[:, 1]) - (tickte[:, 1] - tickte[:, 2]))
        add(f"v4_vol_{c}_slope", slope(varr), slope(varrte))
    ha_wd = monthly(tr, "withdraw_highest_amount")
    ha_wdte = monthly(te, "withdraw_highest_amount")
    add("v4_wd_highest_z", (ha_wd[:, 0] - ha_wd.mean(axis=1)) / np.maximum(ha_wd.std(axis=1), 1.0),
        (ha_wdte[:, 0] - ha_wdte.mean(axis=1)) / np.maximum(ha_wdte.std(axis=1), 1.0))
    add("v4_wd_highest_div_bal", ha_wd[:, 0] / np.maximum(b[:, 0], 1.0),
        ha_wdte[:, 0] / np.maximum(bte[:, 0], 1.0))
    add("v4_zero_vol_months_wd", (vol["withdraw"] == 0).sum(axis=1).astype(float),
        (volte["withdraw"] == 0).sum(axis=1).astype(float))
    add("v4_zero_vol_months_dp", (vol["deposit"] == 0).sum(axis=1).astype(float),
        (volte["deposit"] == 0).sum(axis=1).astype(float))

    # ---- H. balance trajectory part 2 (8)
    qc = np.polyfit(t, b.T, 3)
    qcte = np.polyfit(t, bte.T, 3)
    add("v4_bal_cubic_a", qc[0], qcte[0])
    add("v4_bal_recovery", (b[:, 0] - b.min(axis=1)) / np.maximum(b.max(axis=1) - b.min(axis=1), 1.0),
        (bte[:, 0] - bte.min(axis=1)) / np.maximum(bte.max(axis=1) - bte.min(axis=1), 1.0))
    add("v4_bal_m1_vs_min", b[:, 0] - b.min(axis=1), bte[:, 0] - bte.min(axis=1))
    add("v4_bal_convex", (b[:, 0] - 2 * b[:, 2] + b[:, 5]), (bte[:, 0] - 2 * bte[:, 2] + bte[:, 5]))
    add("v4_outflow_div_balmean", outflow.sum(1) / np.maximum(balmean, 1.0),
        outflowte.sum(1) / np.maximum(balmeante, 1.0))
    add("v4_netflow_div_balmean", net.sum(1) / np.maximum(balmean, 1.0),
        nette.sum(1) / np.maximum(balmeante, 1.0))
    add("v4_bal_slope_x_cv", slope(b) * cv(b), slope(bte) * cv(bte))
    add("v4_earlylate_bal", b[:, 3:].mean(axis=1) / np.maximum(b[:, :3].mean(axis=1), 1.0),
        bte[:, 3:].mean(axis=1) / np.maximum(bte[:, :3].mean(axis=1), 1.0))

    for name in new_tr:
        tr[name] = new_tr[name]
        te[name] = new_te[name]
    names = list(new_tr.keys())
    print(f"v4 battery: {len(names)} candidates", flush=True)
    return tr, te, names

# ---------------------------------------------------------------- Stage 2
def stage2_fe_pass4(tr, te, y, feats359):
    """Greedy forward selection of v4 candidates in gain-ranked batches of 20."""
    print("\n=== Stage 2: FE pass 4 (v4 battery, residual-targeted) ===", flush=True)
    st = load_state()
    if st.get("stage2_done"):
        print("stage2 cached: kept", len(st["stage2_kept"]), "best", st["stage2_best"],
              flush=True)
        return st["stage2_kept"], st["stage2_best"]

    iter6.SEED = 42
    tr4, te4, v4_names = add_features_v4(tr.copy(), te.copy())

    if "ranked_v4" not in st:
        print("gain-ranking fit on full data (359+v4)...", flush=True)
        Xg = tr4[feats359 + v4_names]
        dall = lgb.Dataset(Xg, y, categorical_feature=CAT_COLS)
        m = lgb.train(dict(KITCHEN_SINK, num_threads=4), dall, num_boost_round=800)
        gain = pd.Series(m.feature_importance(importance_type="gain"),
                         index=feats359 + v4_names)
        st["ranked_v4"] = [f for f in gain.sort_values(ascending=False).index
                           if f in v4_names]
        save_state(st)
    ranked_v4 = st["ranked_v4"]

    kept = st.get("stage2_kept", [])
    done_batches = set(st.get("stage2_batches", []))
    if "stage2_best" not in st:
        oof0, _, _ = run_lgbm(tr4[feats359], y, te4[feats359], KITCHEN_SINK)
        st["stage2_best"] = score_oof(y, oof0)
        print(f"stage2 baseline (359, seed42): {st['stage2_best']:.5f}", flush=True)
        save_state(st)
    best = st["stage2_best"]

    batches = [ranked_v4[i:i + BATCH] for i in range(0, len(ranked_v4), BATCH)]
    no_gain = st.get("stage2_nogain", 0)
    for bi, batch in enumerate(batches):
        if bi in done_batches:
            print(f"batch {bi}: skipped (done)", flush=True)
            continue
        if len(feats359) + len(kept) + len(batch) > MAX_FEATURES:
            print(f"batch {bi}: skipped (would exceed {MAX_FEATURES})", flush=True)
            done_batches.add(bi)
            continue
        cand = feats359 + kept + batch
        oof, _, _ = run_lgbm(tr4[cand], y, te4[cand], KITCHEN_SINK)
        sc = score_oof(y, oof)
        ll = log_loss(y, oof)
        auc = roc_auc_score(y, oof)
        improved = sc > best + 1e-6
        if improved:
            kept.extend(batch)
            best = sc
            no_gain = 0
        else:
            no_gain += 1
        print(f"batch {bi} (+{len(batch)} -> {len(cand)} total): "
              f"LL={ll:.5f} AUC={auc:.5f} comb={sc:.5f} "
              f"{'KEPT' if improved else 'dropped'} (best {best:.5f})", flush=True)
        done_batches.add(bi)
        st.update({"stage2_kept": kept, "stage2_batches": sorted(done_batches),
                   "stage2_best": best, "stage2_nogain": no_gain})
        save_state(st)
        if no_gain >= PATIENCE:
            print(f"early stop: {PATIENCE} non-improving batches", flush=True)
            break
    st["stage2_done"] = True
    save_state(st)
    print(f"Stage 2 done: kept {len(kept)} v4 features, best={best:.5f}", flush=True)
    return kept, best


# ---------------------------------------------------------------- Stage 3
def stage3_cb_depth8(tr, te, y, feats):
    """CatBoost depth-8 retest on the (possibly new) feature set, seed 42."""
    print("\n=== Stage 3: CatBoost depth-8 retest (seed 42) ===", flush=True)
    st = load_state()
    if st.get("stage3_done"):
        s = st["stage3"]
        print("stage3 cached:", s["tag"], "hyb=", s["hyb"], "changed=", s["changed"],
              flush=True)
        return s["cfg"], s["changed"]

    X, Xte = tr[feats], te[feats]
    # reference: iter13's seed-42 members if feats unchanged else fresh ref run
    oof_l42 = np.load(os.path.join(OUT, "iter13_lgbm_s42.npz"))["oof"]
    oof_c42_ref = np.load(os.path.join(OUT, "iter13_catboost_s42.npz"))["oof"]
    base_hyb = score_oof(y, hybrid(oof_l42, oof_c42_ref, 0.5, 0.5))
    print(f"ref CB(d6) s42 hybrid(wg.5/wl.5)={base_hyb:.5f}", flush=True)

    cfg8 = dict(CB_REF, depth=8)
    oof8, _ = iter13.run_catboost_cfg(X, y, Xte, cfg8, "cb_d8_retest")
    sc_hyb8 = score_oof(y, hybrid(oof_l42, oof8, 0.5, 0.5))
    changed = bool(sc_hyb8 > base_hyb + 1e-6)
    s = {"tag": "cb_d8_retest", "hyb": sc_hyb8, "base_hyb": base_hyb,
         "cfg": cfg8 if changed else CB_REF, "changed": changed}
    st["stage3_done"] = True
    st["stage3"] = s
    save_state(st)
    print(f"Stage 3: depth-8 hyb={sc_hyb8:.5f} vs ref {base_hyb:.5f} -> "
          f"{'KEPT' if changed else 'rejected'}", flush=True)
    return s["cfg"], changed


# ---------------------------------------------------------------- Stage 4
def residual_stack_screen(y, p_oof, X, tag="screen"):
    """Honest pseudo-residual stacking screen: small LGBM on r=y-p_oof with its
    own 5-fold CV; tune correction weight alpha on OOF. Returns
    (alpha, resid_oof, score) or (0.0, None, base_score) if no gain."""
    base = score_oof(y, p_oof)
    r = y - p_oof
    oof_r = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    prm = dict(KITCHEN_SINK, num_leaves=15, min_data_in_leaf=100,
               learning_rate=0.05, feature_fraction=0.7, bagging_fraction=0.8,
               bagging_freq=1, lambda_l1=1.0, lambda_l2=5.0)
    for trn, val in skf.split(X, y):
        dtr = lgb.Dataset(X.iloc[trn], r[trn])
        m = lgb.train(prm, dtr, num_boost_round=500)
        oof_r[val] = m.predict(X.iloc[val])
    best_a, best_sc = 0.0, base
    for a in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
        sc = score_oof(y, np.clip(p_oof + a * oof_r, EPS, 1 - EPS))
        if sc > best_sc + 1e-6:
            best_a, best_sc = a, sc
    print(f"  [{tag}] residual-stack: base={base:.5f} best_alpha={best_a} "
          f"sc={best_sc:.5f} ({'GAIN' if best_a else 'no gain'})", flush=True)
    return best_a, (oof_r if best_a else None), best_sc


def train_resid_full(X, r, Xte, alpha):
    """Train residual model on full data; return test residual predictions."""
    prm = dict(KITCHEN_SINK, num_leaves=15, min_data_in_leaf=100,
               learning_rate=0.05, feature_fraction=0.7, bagging_fraction=0.8,
               bagging_freq=1, lambda_l1=1.0, lambda_l2=5.0)
    dtr = lgb.Dataset(X, r)
    m = lgb.train(prm, dtr, num_boost_round=500)
    return m.predict(Xte)


# ---------------------------------------------------------------- Stage 5/6 trainers
def train_lgbm_seed(seed, tr0, te0, y, feats, tag_prefix="iter14"):
    iter6.SEED = seed
    p = os.path.join(OUT, f"{tag_prefix}_lgbm_s{seed}.npz")
    if os.path.exists(p):
        d = np.load(p)
        print(f"seed {seed}: LGBM loaded cached", flush=True)
        return d["oof"], d["test"]
    tr = tr0.copy()
    te = te0.copy()
    tr, te, _ = add_target_encoding(tr, te, y)
    prm = dict(KITCHEN_SINK, seed=seed)
    print(f"seed {seed}: training LGBM...", flush=True)
    oof, test, mi = run_lgbm(tr[feats], y, te[feats], prm)
    np.savez(p, oof=oof, test=test, iters=mi)
    print(f"seed {seed}: LGBM done, mean_iter={mi:.0f}, comb={score_oof(y, oof):.5f}",
          flush=True)
    return oof, test


def train_cb_seed(seed, tr0, te0, y, feats, cb_cfg, tag_prefix="iter14"):
    iter8.SEED = seed
    p = os.path.join(OUT, f"{tag_prefix}_catboost_s{seed}.npz")
    if os.path.exists(p):
        d = np.load(p)
        print(f"seed {seed}: CatBoost loaded cached", flush=True)
        return d["oof"], d["test"]
    iter6.SEED = seed
    tr = tr0.copy()
    te = te0.copy()
    tr, te, _ = add_target_encoding(tr, te, y)
    X, Xte = tr[feats], te[feats]
    print(f"seed {seed}: training CatBoost {cb_cfg}...", flush=True)
    oof = np.zeros(len(X))
    test = np.zeros(len(Xte))
    iters = []
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for fold, (trn, val) in enumerate(skf.split(X, y)):
        Xtr = X.iloc[trn].copy()
        Xva = X.iloc[val].copy()
        Xte_c = Xte.copy()
        for c in CAT_ORDER:
            Xtr[c] = Xtr[c].astype(str)
            Xva[c] = Xva[c].astype(str)
            Xte_c[c] = Xte[c].astype(str)
        m = CatBoostClassifier(iterations=cb_cfg["iterations"],
                               learning_rate=cb_cfg["learning_rate"],
                               depth=cb_cfg["depth"],
                               l2_leaf_reg=cb_cfg["l2_leaf_reg"],
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
    np.savez(p, oof=oof, test=test, iters=np.mean(iters))
    print(f"seed {seed}: CatBoost done, mean_iter={np.mean(iters):.0f}, "
          f"comb={score_oof(y, oof):.5f}", flush=True)
    return oof, test


def save_submission_csv(test_pred, te_ids, tag, score, spec):
    test_pred = np.clip(test_pred, 1e-6, 1 - 1e-6)
    path = os.path.join(OUT, f"{tag}.csv")
    pd.DataFrame({ID_COL: te_ids, "Target": test_pred}).to_csv(path, index=False)
    print(f"Saved submissions/{tag}.csv (test mean {test_pred.mean():.4f}, "
          f"OOF comb {score:.5f})", flush=True)
    with open(os.path.join(OUT, f"{tag}_results.txt"), "w") as f:
        f.write(f"ITERATION 14 ({tag}) results\nbar (iter13 6-seed): {BAR}\n")
        f.write(f"FINAL OOF combined={score:.5f} BEATS {BAR}\n")
        f.write(f"spec: {json.dumps(spec)}\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="2,3,4,5,6",
                    help="comma-separated subset of stages to run")
    args = ap.parse_args()
    stages = set(args.stages.split(","))
    os.makedirs(OUT, exist_ok=True)
    st = load_state()

    # ---- base 359 set (seed 42) ----
    tr359, te359, y, feats359, bal_p25 = build_359(seed=42)
    te_ids = te359[ID_COL].values

    # ---- Stage 2: v4 FE pass ----
    kept_v4 = []
    if "2" in stages:
        kept_v4, best2 = stage2_fe_pass4(tr359, te359, y, feats359)
    elif st.get("stage2_done"):
        kept_v4 = st["stage2_kept"]
    feats_final = feats359 + kept_v4
    feats_changed = len(kept_v4) > 0
    print(f"\nfeats_final={len(feats_final)} (v4 kept: {len(kept_v4)})", flush=True)

    tr0, te0, _ = add_features_v4(tr359.copy(), te359.copy())

    # ---- Stage 3: CatBoost depth-8 retest ----
    cb_cfg, cb_changed = (CB_REF, False)
    if "3" in stages:
        cb_cfg, cb_changed = stage3_cb_depth8(tr0, te0, y, feats_final)
    elif st.get("stage3_done"):
        s = st["stage3"]
        cb_cfg, cb_changed = s["cfg"], s["changed"]
    print(f"cb_cfg={cb_cfg} cb_changed={cb_changed}", flush=True)

    # ---- Stage 4: residual-stack screen on seed-42 hybrid ----
    resid_alpha, resid_wins = 0.0, False
    if "4" in stages:
        print("\n=== Stage 4: pseudo-residual stacking screen ===", flush=True)
        if not st.get("stage4_done"):
            oof_l42 = np.load(os.path.join(OUT, "iter13_lgbm_s42.npz"))["oof"]
            oof_c42 = np.load(os.path.join(OUT, "iter13_catboost_s42.npz"))["oof"]
            p42 = hybrid(oof_l42, oof_c42, 0.5, 0.5)
            a, _, sc = residual_stack_screen(y, p42, tr0[feats_final], "s42")
            st["stage4"] = {"alpha": a, "score": sc,
                            "base": score_oof(y, p42)}
            st["stage4_done"] = True
            save_state(st)
        resid_alpha = st["stage4"]["alpha"]
        resid_wins = resid_alpha > 0
        print(f"Stage 4 screen: alpha={resid_alpha} wins={resid_wins}", flush=True)
    elif st.get("stage4_done"):
        resid_alpha = st["stage4"]["alpha"]
        resid_wins = resid_alpha > 0

    # ---- Stage 5: seed assembly (6 -> 10 if the base changed) ----
    saved = []
    if "5" in stages or "6" in stages:
        go10 = feats_changed or cb_changed
        seeds = SEEDS10 if go10 else SEEDS6
        print(f"\n=== Stage 5/6: seed assembly ({len(seeds)} seeds; "
              f"{'10-seed expansion' if go10 else '6-seed, base unchanged'}) ===",
              flush=True)
        need_retrain = feats_changed or cb_changed
        tag_prefix = "iter14"
        oofs_l, tests_l, oofs_c, tests_c = [], [], [], []
        for s in seeds:
            if need_retrain or s not in SEEDS6:
                oof_l, test_l = train_lgbm_seed(s, tr0, te0, y, feats_final,
                                               tag_prefix)
            else:
                d = np.load(os.path.join(OUT, f"iter13_lgbm_s{s}.npz"))
                oof_l, test_l = d["oof"], d["test"]
                print(f"seed {s}: LGBM reused iter13 cache", flush=True)
            oofs_l.append(oof_l)
            tests_l.append(test_l)
            if need_retrain or cb_changed or s not in SEEDS6:
                oof_c, test_c = train_cb_seed(s, tr0, te0, y, feats_final,
                                             cb_cfg, tag_prefix)
            else:
                d = np.load(os.path.join(OUT, f"iter13_catboost_s{s}.npz"))
                oof_c, test_c = d["oof"], d["test"]
                print(f"seed {s}: CatBoost reused iter13 cache", flush=True)
            oofs_c.append(oof_c)
            tests_c.append(test_c)

        nm, oof_b, test_b, sc_b = blend_bakeoff(y, oofs_l, tests_l, oofs_c,
                                               tests_c, seeds, "iter14-final")
        best_nm, best_oof, best_test, best_sc = nm, oof_b, test_b, sc_b
        best_kind = f"hybrid:{nm}"

        # residual correction on the FINAL blend (honest: own-CV residual OOF)
        a_fin, _, sc_res = residual_stack_screen(y, oof_b, tr0[feats_final],
                                                "final-blend")
        if a_fin > 0:
            r_full = y - oof_b
            r_test = train_resid_full(tr0[feats_final], r_full, te0[feats_final],
                                     a_fin)
            best_test = np.clip(test_b + a_fin * r_test, EPS, 1 - EPS)
            best_sc = sc_res
            best_kind = f"hybrid:{nm}+resid(a={a_fin})"
            print(f"residual correction APPLIED on final blend", flush=True)

        # LR stacker on the FINAL member OOFs (honest CV)
        meta_oof = np.column_stack(oofs_l + oofs_c)
        lr = LogisticRegressionCV(Cs=8, cv=5, max_iter=5000, n_jobs=4)
        meta_pred = cross_val_predict(lr, meta_oof, y, cv=5,
                                      method="predict_proba")[:, 1]
        sc_lr = score_oof(y, np.clip(meta_pred, EPS, 1 - EPS))
        print(f"LR stacker (final members) honest-OOF={sc_lr:.5f} "
              f"vs best={best_sc:.5f}", flush=True)
        if sc_lr > best_sc + 1e-6:
            lr.fit(meta_oof, y)
            meta_test = np.column_stack(tests_l + tests_c)
            best_test = np.clip(lr.predict_proba(meta_test)[:, 1], EPS, 1 - EPS)
            best_oof = np.clip(meta_pred, EPS, 1 - EPS)
            best_sc = sc_lr
            best_kind = "lr_stacker"
            print("LR stacker WINS the final blend", flush=True)

        print(f"\nFinal: {best_kind} OOF comb={best_sc:.5f} "
              f"({'BEATS' if best_sc > BAR else 'below'} bar {BAR})", flush=True)
        if best_sc > BAR:
            spec = {"kind": best_kind, "blend": nm, "seeds": seeds,
                    "feats": len(feats_final), "v4_kept": kept_v4,
                    "cb_cfg": cb_cfg, "cb_changed": cb_changed,
                    "resid_alpha": a_fin, "n10": len(seeds) == 10}
            p = save_submission_csv(best_test, te_ids, "iter14", best_sc, spec)
            saved.append(p)
            st["final"] = {"score": best_sc, "spec": spec, "file": p}
            save_state(st)
        else:
            print("No candidate beat the bar; no CSV written.", flush=True)

    print("EXIT:0")


if __name__ == "__main__":
    main()
