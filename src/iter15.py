"""Iteration 15: v5 residual-pocket battery -> 6-seed LGBM+CatBoost -> blend re-tune.
Fallback: MLP diversity screen.

Bar: iter14 10-seed hybrid A_wg0.5_wl0.5 = 0.82543 (399 feats, public 0.72304).
Leader 0.74293, gap 0.0199. 9 submission slots left today. User: #1 TODAY.

Residual pockets (from iter14 analysis): low-activity users worst slice,
78.5% of worst-decile errors are false negatives (under-predict stress),
missed positives have decent m1 balances (hidden stress), HVC/LVC
overrepresented in errors.

RULES: 5-fold stratified CV seed 42, per-fold isotonic. NEVER submits
(parent handles browser). Resume-safe via submissions/iter15_state.json.
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iter6
import iter8
import iter12
import iter13
import iter14
from iter6 import (add_target_encoding, run_lgbm, KITCHEN_SINK, CAT_COLS,
                   N_SPLITS, TARGET, ID_COL, DATA, OUT, EPS, combined)
from iter8 import score_oof, CAT_ORDER
from iter12 import hybrid, blend_bakeoff

BAR = 0.82543          # iter14 10-seed hybrid
SEEDS6 = [42, 7, 2025, 5, 11, 777]
BATCH = 20
PATIENCE = 2
MAX_FEATURES = 445
STATE15 = os.path.join(OUT, "iter15_state.json")
CB_D8 = {"depth": 8, "learning_rate": 0.03, "iterations": 2000,
         "l2_leaf_reg": 3.0}   # won iter14's retest


def save_state(st):
    with open(STATE15, "w") as f:
        json.dump(st, f)


def load_state():
    if os.path.exists(STATE15):
        with open(STATE15) as f:
            return json.load(f)
    return {}


def build_399(seed=42):
    """iter14's final 399-feature set (359 + 40 kept v4)."""
    tr359, te359, y, feats359, bal_p25 = iter14.build_359(seed=seed)
    with open(os.path.join(OUT, "iter14_state.json")) as f:
        kept_v4 = json.load(f)["stage2_kept"]
    tr, te, _ = iter14.add_features_v4(tr359.copy(), te359.copy())
    feats = feats359 + kept_v4
    missing = [c for c in feats if c not in tr.columns]
    assert not missing, f"missing feats: {missing[:5]}"
    print(f"399-feat set built (seed {seed}): {len(feats)} feats", flush=True)
    return tr, te, y, feats, bal_p25


# ---------------------------------------------------------------- v5 battery
def add_features_v5(tr, te):
    """Fifth battery (~82 candidates, v5_ prefix), doubling down on residual
    pockets: low-activity deep interactions, hidden-stress second-order,
    HVC/LVC-conditioned tails, plus new angles. All unsupervised."""
    new_tr, new_te = {}, {}

    def add(name, v_tr, v_te):
        if name in tr.columns or name in new_tr:
            raise RuntimeError(f"name collision: {name}")
        a = np.asarray(v_tr, dtype=float)
        b = np.asarray(v_te, dtype=float)
        a[~np.isfinite(a)] = 0.0
        b[~np.isfinite(b)] = 0.0
        new_tr[name] = a
        new_te[name] = b

    def monthly(df, metric):
        return df[[f"m{i}_{metric}" for i in range(1, 7)]].values.astype(float)

    b = monthly(tr, "daily_avg_bal")
    bte = monthly(te, "daily_avg_bal")
    t = np.arange(6, dtype=float)
    tc = t - t.mean()

    def slope(a):
        return (a * tc).sum(axis=1) / (tc ** 2).sum()

    def accel(a):
        return (a[:, 0] - a[:, 1]) - (a[:, 1] - a[:, 2])

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

    net_slope = slope(net)
    net_slopete = slope(nette)
    bal_drop = (b.max(axis=1) - b[:, 0]) / np.maximum(b.max(axis=1), 1.0)
    bal_dropte = (bte.max(axis=1) - bte[:, 0]) / np.maximum(bte.max(axis=1), 1.0)
    wd_pressure = wd[:, 0] / np.maximum(b[:, 0], 1.0)
    wd_pressurete = wdte[:, 0] / np.maximum(bte[:, 0], 1.0)
    runway_m1 = b[:, 0] / np.maximum(outflow[:, 0], 1.0)
    runway_m1te = bte[:, 0] / np.maximum(outflowte[:, 0], 1.0)

    # ---- A. low-activity deep interactions (~22): worst slice
    inact = 1.0 - act
    inactte = 1.0 - actte
    inact2 = inact ** 2
    inact2te = inactte ** 2
    low20 = (act < 0.2).astype(float)
    low20te = (actte < 0.2).astype(float)
    signflips = (np.diff(np.sign(net), axis=1) != 0).sum(axis=1).astype(float)
    signflipste = (np.diff(np.sign(nette), axis=1) != 0).sum(axis=1).astype(float)
    net_min = net.min(axis=1)
    net_minte = nette.min(axis=1)
    sr = (inflow - outflow) / np.maximum(inflow, 1.0)
    srte = (inflowte - outflowte) / np.maximum(inflowte, 1.0)
    sr_min = sr.min(axis=1)
    sr_minte = srte.min(axis=1)
    wd_mean = wd.mean(axis=1)
    wd_meante = wdte.mean(axis=1)
    wd_spike = wd[:, 0] / np.maximum(wd_mean, 1.0)
    wd_spikete = wdte[:, 0] / np.maximum(wd_meante, 1.0)
    dp_collapse = dp[:, 0] / np.maximum(dp.mean(axis=1), 1.0)
    dp_collapsete = dpte[:, 0] / np.maximum(dpte.mean(axis=1), 1.0)
    add("v5_inact_x_netslope", inact * net_slope, inactte * net_slopete)
    add("v5_inact_x_baldrop", inact * bal_drop, inactte * bal_dropte)
    add("v5_inact_x_wdpressure", inact * wd_pressure, inactte * wd_pressurete)
    add("v5_inact_x_runway", inact * runway_m1, inactte * runway_m1te)
    add("v5_inact_x_signflips", inact * signflips, inactte * signflipste)
    add("v5_inact_x_outflowcv", inact * cv(outflow), inactte * cv(outflowte))
    add("v5_inact_x_netmin", inact * net_min / np.maximum(balmean, 1.0),
        inactte * net_minte / np.maximum(balmeante, 1.0))
    add("v5_inact_x_savingsmin", inact * sr_min, inactte * sr_minte)
    add("v5_inact2_x_netslope", inact2 * net_slope, inact2te * net_slopete)
    add("v5_inact2_x_wdpressure", inact2 * wd_pressure, inact2te * wd_pressurete)
    add("v5_low20_x_netslope", low20 * net_slope, low20te * net_slopete)
    add("v5_low20_x_wdspike", low20 * wd_spike, low20te * wd_spikete)
    add("v5_low20_x_dpcollapse", low20 * dp_collapse, low20te * dp_collapsete)
    add("v5_low20_x_savingsmin", low20 * sr_min, low20te * sr_minte)
    add("v5_low20_x_baldrop", low20 * bal_drop, low20te * bal_dropte)
    add("v5_act_x_netflowz",
        act * (net[:, 0] - net.mean(axis=1)) / np.maximum(net.std(axis=1), 1.0),
        actte * (nette[:, 0] - nette.mean(axis=1)) / np.maximum(nette.std(axis=1), 1.0))
    add("v5_act_x_outflowz",
        act * (outflow[:, 0] - outflow.mean(axis=1)) / np.maximum(outflow.std(axis=1), 1.0),
        actte * (outflowte[:, 0] - outflowte.mean(axis=1)) / np.maximum(outflowte.std(axis=1), 1.0))
    add("v5_act_decile", pd.qcut(act, 10, labels=False, duplicates="drop").astype(float),
        pd.qcut(actte, 10, labels=False, duplicates="drop").astype(float))
    add("v5_inact_x_wdaccel", inact * accel(wd), inactte * accel(wdte))
    add("v5_inact_x_dpaccel", inact * accel(dp), inactte * accel(dpte))
    add("v5_lowact_seg_LVC", low20 * seg_masks["LVC"], low20te * seg_masks_te["LVC"])
    add("v5_lowact_seg_HVC", low20 * seg_masks["HVC"], low20te * seg_masks_te["HVC"])

    # ---- B. hidden-stress second-order (~22): decent balance, bad flows
    bal_med = np.median(b[:, 0])
    bal_medte = np.median(bte[:, 0])
    ok_bal = (b[:, 0] > bal_med).astype(float)
    ok_balte = (bte[:, 0] > bal_medte).astype(float)
    bal_stable = (cv(b) < np.median(cv(b))).astype(float)
    bal_stablete = (cv(bte) < np.median(cv(bte))).astype(float)
    wd_accel = accel(wd)
    wd_accelte = accel(wdte)
    out_accel = accel(outflow)
    out_accelte = accel(outflowte)
    add("v5_wdaccel_x_balstable", wd_accel * bal_stable, wd_accelte * bal_stablete)
    add("v5_outaccel_x_okbal", out_accel * ok_bal, out_accelte * ok_balte)
    add("v5_wdaccel_x_okbal", wd_accel * ok_bal, wd_accelte * ok_balte)
    dp_reg = 1.0 / (1.0 + cv(dp))
    dp_regte = 1.0 / (1.0 + cv(dpte))
    add("v5_dpreg_x_collapse", dp_reg * (1 - dp_collapse), dp_regte * (1 - dp_collapsete))
    recv_share = tv["received"] / np.maximum(inflow, 1.0)
    recv_sharete = tvte["received"] / np.maximum(inflowte, 1.0)
    add("v5_recvshare_accel", accel(recv_share), accel(recv_sharete))
    add("v5_net_deterioration", net[:, 0] - net[:, 5], nette[:, 0] - nette[:, 5])
    add("v5_sr_traj", sr[:, 0] - sr.mean(axis=1), srte[:, 0] - srte.mean(axis=1))
    add("v5_bal_velocity_2m", b[:, 0] - b[:, 2], bte[:, 0] - bte[:, 2])
    negm = (net < 0)
    negmte = (nette < 0)
    add("v5_quiet_drain",
        negm.sum(axis=1) * np.abs(net).mean(axis=1) / np.maximum(balmean, 1.0),
        negmte.sum(axis=1) * np.abs(nette).mean(axis=1) / np.maximum(balmeante, 1.0))
    add("v5_inflow_down_accel", np.maximum(0, -accel(inflow)),
        np.maximum(0, -accel(inflowte)))
    add("v5_wd_up_accel", np.maximum(0, wd_accel), np.maximum(0, wd_accelte))
    add("v5_okbal_x_netaccelneg",
        ok_bal * np.minimum(accel(net), 0), ok_balte * np.minimum(accel(nette), 0))
    add("v5_netaccel", accel(net), accel(nette))
    add("v5_balaccel", accel(b), accel(bte))
    add("v5_sr_accel", accel(sr), accel(srte))
    add("v5_okbal_x_wdaccel", ok_bal * wd_accel, ok_balte * wd_accelte)
    add("v5_dp_vol_zero_x_cv",
        (vol["deposit"] == 0).sum(axis=1) * cv(dp),
        (volte["deposit"] == 0).sum(axis=1) * cv(dpte))
    add("v5_netflow_late_early",
        net[:, :2].mean(axis=1) - net[:, 4:].mean(axis=1),
        nette[:, :2].mean(axis=1) - nette[:, 4:].mean(axis=1))
    add("v5_outflow_late_early",
        outflow[:, :2].mean(axis=1) / np.maximum(outflow[:, 4:].mean(axis=1), 1.0),
        outflowte[:, :2].mean(axis=1) / np.maximum(outflowte[:, 4:].mean(axis=1), 1.0))
    add("v5_wd_m1_vs_med", wd[:, 0] / np.maximum(np.median(wd, axis=1), 1.0),
        wdte[:, 0] / np.maximum(np.median(wdte, axis=1), 1.0))
    add("v5_dp_m1_vs_med", dp[:, 0] / np.maximum(np.median(dp, axis=1), 1.0),
        dpte[:, 0] / np.maximum(np.median(dpte, axis=1), 1.0))

    # ---- C. HVC/LVC-conditioned tails (~20)
    maxdd = (np.maximum.accumulate(b[:, ::-1], axis=1) - b[:, ::-1]).max(axis=1)
    maxddte = (np.maximum.accumulate(bte[:, ::-1], axis=1) - bte[:, ::-1]).max(axis=1)
    neg = (net < 0).astype(int)
    negte = (nette < 0).astype(int)
    s = np.zeros(len(tr)); st_ = np.zeros(len(te))
    for i in range(6):
        s = np.where((neg[:, i] == 1) & (s == i), i + 1, s)
        st_ = np.where((negte[:, i] == 1) & (st_ == i), i + 1, st_)
    worst_nf = net.min(axis=1) / np.maximum(balmean, 1.0)
    worst_nfte = nette.min(axis=1) / np.maximum(balmeante, 1.0)
    out_z = (outflow[:, 0] - outflow.mean(axis=1)) / np.maximum(outflow.std(axis=1), 1.0)
    out_zte = (outflowte[:, 0] - outflowte.mean(axis=1)) / np.maximum(outflowte.std(axis=1), 1.0)
    ha_wd = monthly(tr, "withdraw_highest_amount")
    ha_wdte = monthly(te, "withdraw_highest_amount")
    wdhi_bal = ha_wd[:, 0] / np.maximum(b[:, 0], 1.0)
    wdhi_balte = ha_wdte[:, 0] / np.maximum(bte[:, 0], 1.0)
    net_z = (net[:, 0] - net.mean(axis=1)) / np.maximum(net.std(axis=1), 1.0)
    net_zte = (nette[:, 0] - nette.mean(axis=1)) / np.maximum(nette.std(axis=1), 1.0)
    dcl = (np.diff(b[:, ::-1], axis=1) < 0).astype(int)
    dclte = (np.diff(bte[:, ::-1], axis=1) < 0).astype(int)
    def longest_run(a):
        best = np.zeros(a.shape[0]); cur = np.zeros(a.shape[0])
        for i in range(a.shape[1]):
            cur = np.where(a[:, i] == 1, cur + 1, 0)
            best = np.maximum(best, cur)
        return best
    bal_run = longest_run(dcl); bal_runte = longest_run(dclte)
    savneg = np.minimum(sr.min(axis=1), 0); savnegte = np.minimum(srte.min(axis=1), 0)
    tail_feats = {
        "maxdd": (maxdd, maxddte), "negstreak": (s, st_),
        "worstnf": (worst_nf, worst_nfte), "outz": (out_z, out_zte),
        "wdhi": (wdhi_bal, wdhi_balte), "netz": (net_z, net_zte),
        "balrun": (bal_run, bal_runte), "savneg": (savneg, savnegte),
    }
    for sg in ("HVC", "LVC"):
        for fname, (arr, arrte) in tail_feats.items():
            add(f"v5_seg{sg}_x_{fname}", seg_masks[sg] * arr, seg_masks_te[sg] * arrte)
    arpu_q = pd.qcut(arpu, 5, labels=False, duplicates="drop").astype(float)
    arpu_qte = pd.qcut(arpute, 5, labels=False, duplicates="drop").astype(float)
    add("v5_segLVC_x_arpuq", seg_masks["LVC"] * arpu_q, seg_masks_te["LVC"] * arpu_qte)
    add("v5_segHVC_x_arpuq", seg_masks["HVC"] * arpu_q, seg_masks_te["HVC"] * arpu_qte)
    add("v5_segLVC_x_inact", seg_masks["LVC"] * inact, seg_masks_te["LVC"] * inactte)
    add("v5_segHVC_x_inact", seg_masks["HVC"] * inact, seg_masks_te["HVC"] * inactte)

    # ---- D. new angles (~18): channel substitution, robust z, velocity
    pb_slope = slope(tv["paybill"])
    pb_slopete = slope(tvte["paybill"])
    ms_slope = slope(tv["mm_send"])
    ms_slopete = slope(tvte["mm_send"])
    add("v5_channel_subst",
        np.maximum(0, -pb_slope) * np.maximum(0, ms_slope),
        np.maximum(0, -pb_slopete) * np.maximum(0, ms_slopete))
    add("v5_pb_ms_diverge",
        np.abs(pb_slope - ms_slope) / (np.abs(pb_slope) + np.abs(ms_slope) + 1e-9),
        np.abs(pb_slopete - ms_slopete) / (np.abs(pb_slopete) + np.abs(ms_slopete) + 1e-9))

    def mad(a):
        med = np.median(a, axis=1, keepdims=True)
        return np.median(np.abs(a - med), axis=1)

    def robust_z(a):
        med = np.median(a, axis=1)
        m = mad(a)
        return (a[:, 0] - med) / np.maximum(m, 1e-9)

    add("v5_outflow_robustz", robust_z(outflow), robust_z(outflowte))
    add("v5_bal_robustz", robust_z(b), robust_z(bte))
    add("v5_net_robustz", robust_z(net), robust_z(nette))
    add("v5_wd_robustz", robust_z(wd), robust_z(wdte))
    add("v5_outflow_m1_med", outflow[:, 0] / np.maximum(np.median(outflow, axis=1), 1.0),
        outflowte[:, 0] / np.maximum(np.median(outflowte, axis=1), 1.0))
    add("v5_net_m1_med", net[:, 0] / (np.abs(np.median(net, axis=1)) + 1.0),
        nette[:, 0] / (np.abs(np.median(nette, axis=1)) + 1.0))
    ha_wd = monthly(tr, "withdraw_highest_amount")
    ha_wdte = monthly(te, "withdraw_highest_amount")
    add("v5_wdhi_accel", accel(ha_wd), accel(ha_wdte))
    add("v5_dpvol0_x_dpmean",
        (vol["deposit"] == 0).sum(axis=1).astype(float) * dp.mean(axis=1) / np.maximum(balmean, 1.0),
        (volte["deposit"] == 0).sum(axis=1).astype(float) * dpte.mean(axis=1) / np.maximum(balmeante, 1.0))
    add("v5_age_x_inact", age * inact, agete * inactte)
    add("v5_age_x_okbalneg",
        age * ok_bal * np.minimum(net_slope, 0),
        agete * ok_balte * np.minimum(net_slopete, 0))
    add("v5_young_x_wdaccel",
        (age < 30).astype(float) * wd_accel, (agete < 30).astype(float) * wd_accelte)
    add("v5_recv_vol_trend", slope(vol["received"]), slope(volte["received"]))
    add("v5_mmsend_share", tv["mm_send"][:, 0] / np.maximum(outflow[:, 0], 1.0),
        tvte["mm_send"][:, 0] / np.maximum(outflowte[:, 0], 1.0))
    add("v5_paybill_share", tv["paybill"][:, 0] / np.maximum(outflow[:, 0], 1.0),
        tvte["paybill"][:, 0] / np.maximum(outflowte[:, 0], 1.0))

    for name in new_tr:
        tr[name] = new_tr[name]
        te[name] = new_te[name]
    names = list(new_tr.keys())
    assert all(np.isfinite(tr[n]).all() and np.isfinite(te[n]).all() for n in names)
    print(f"v5 battery: {len(names)} candidates (all finite)", flush=True)
    return tr, te, names


# ---------------------------------------------------------------- Stage 1
def stage_greedy_v5(tr, te, y, feats399):
    """Greedy forward selection of v5 candidates in gain-ranked batches of 20."""
    print("\n=== Stage 1: FE pass 5 (v5 battery, residual-pocket double-down) ===",
          flush=True)
    st = load_state()
    if st.get("stage1_done"):
        print("stage1 cached: kept", len(st["stage1_kept"]), "best",
              st["stage1_best"], flush=True)
        return st["stage1_kept"], st["stage1_best"]

    iter6.SEED = 42
    tr5, te5, v5_names = add_features_v5(tr.copy(), te.copy())

    if "ranked_v5" not in st:
        print("gain-ranking fit on full data (399+v5)...", flush=True)
        Xg = tr5[feats399 + v5_names]
        dall = lgb.Dataset(Xg, y, categorical_feature=CAT_COLS)
        m = lgb.train(dict(KITCHEN_SINK, num_threads=4), dall, num_boost_round=800)
        gain = pd.Series(m.feature_importance(importance_type="gain"),
                         index=feats399 + v5_names)
        st["ranked_v5"] = [f for f in gain.sort_values(ascending=False).index
                           if f in v5_names]
        save_state(st)
    ranked_v5 = st["ranked_v5"]

    kept = st.get("stage1_kept", [])
    done_batches = set(st.get("stage1_batches", []))
    if "stage1_best" not in st:
        oof0, _, _ = run_lgbm(tr5[feats399], y, te5[feats399], KITCHEN_SINK)
        st["stage1_best"] = score_oof(y, oof0)
        print(f"stage1 baseline (399, seed42): {st['stage1_best']:.5f}", flush=True)
        save_state(st)
    best = st["stage1_best"]

    batches = [ranked_v5[i:i + BATCH] for i in range(0, len(ranked_v5), BATCH)]
    no_gain = st.get("stage1_nogain", 0)
    for bi, batch in enumerate(batches):
        if bi in done_batches:
            print(f"batch {bi}: skipped (done)", flush=True)
            continue
        if len(feats399) + len(kept) + len(batch) > MAX_FEATURES:
            print(f"batch {bi}: skipped (would exceed {MAX_FEATURES})", flush=True)
            done_batches.add(bi)
            continue
        cand = feats399 + kept + batch
        oof, _, _ = run_lgbm(tr5[cand], y, te5[cand], KITCHEN_SINK)
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
        st.update({"stage1_kept": kept, "stage1_batches": sorted(done_batches),
                   "stage1_best": best, "stage1_nogain": no_gain})
        save_state(st)
        if no_gain >= PATIENCE:
            print(f"early stop: {PATIENCE} non-improving batches", flush=True)
            break
    st["stage1_done"] = True
    save_state(st)
    print(f"Stage 1 done: kept {len(kept)} v5 features, best={best:.5f}", flush=True)
    return kept, best


# ---------------------------------------------------------------- Stage 2
def stage_6seed_blend(tr0, te0, y, feats_final, te_ids, tag="iter15"):
    """6-seed LGBM+CatBoost(d8) + hybrid blend bake-off. Saves CSV if > BAR."""
    print(f"\n=== Stage 2: 6-seed assembly + blend ({len(feats_final)} feats) ===",
          flush=True)
    oofs_l, tests_l, oofs_c, tests_c = [], [], [], []
    for s in SEEDS6:
        oof_l, test_l = iter14.train_lgbm_seed(s, tr0, te0, y, feats_final,
                                              tag_prefix=tag)
        oof_c, test_c = iter14.train_cb_seed(s, tr0, te0, y, feats_final,
                                            CB_D8, tag_prefix=tag)
        oofs_l.append(oof_l)
        tests_l.append(test_l)
        oofs_c.append(oof_c)
        tests_c.append(test_c)
    nm, oof_b, test_b, sc_b = blend_bakeoff(y, oofs_l, tests_l, oofs_c,
                                           tests_c, SEEDS6, f"{tag}-final")
    print(f"\nStage 2 final: hybrid:{nm} OOF comb={sc_b:.5f} "
          f"({'BEATS' if sc_b > BAR else 'below'} bar {BAR})", flush=True)
    path = None
    if sc_b > BAR:
        spec = {"kind": f"hybrid:{nm}", "blend": nm, "seeds": SEEDS6,
                "feats": len(feats_final), "cb_cfg": CB_D8}
        test_pred = np.clip(test_b, 1e-6, 1 - 1e-6)
        path = os.path.join(OUT, f"{tag}.csv")
        pd.DataFrame({ID_COL: te_ids, "Target": test_pred}).to_csv(path,
                                                                  index=False)
        print(f"Saved submissions/{tag}.csv (test mean {test_pred.mean():.4f}, "
              f"OOF comb {sc_b:.5f})", flush=True)
        with open(os.path.join(OUT, f"{tag}_results.txt"), "w") as f:
            f.write(f"ITERATION 15 ({tag}) results\nbar (iter14 10-seed): {BAR}\n")
            f.write(f"FINAL OOF combined={sc_b:.5f} BEATS {BAR}\n")
            f.write(f"spec: {json.dumps(spec)}\n")
    return sc_b, path


# ---------------------------------------------------------------- Stage 3 (fallback)
def stage_mlp_screen(y, tr, te, feats, oof_blend, test_blend, te_ids, tag):
    """MLP diversity screen: 5-fold CV MLP + per-fold isotonic, then blend
    its OOF with the stage-2 blend at small weights. Saves CSV if it wins."""
    print("\n=== Stage 3 (fallback): MLP diversity screen ===", flush=True)
    st = load_state()
    if st.get("stage3_done"):
        print("stage3 cached: score", st["stage3"].get("score"), flush=True)
        return st["stage3"].get("score", 0.0), st["stage3"].get("path")

    X = tr[feats].values.astype(np.float32)
    Xte = te[feats].values.astype(np.float32)
    # scale using train only
    oof_m = np.zeros(len(X))
    test_m = np.zeros(len(Xte))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    for fold, (trn, val) in enumerate(skf.split(X, y)):
        sc = StandardScaler()
        Ztr = sc.fit_transform(X[trn])
        Zva = sc.transform(X[val])
        Zte = sc.transform(Xte)
        clf = MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-4,
                            learning_rate_init=1e-3, max_iter=400,
                            early_stopping=True, validation_fraction=0.1,
                            n_iter_no_change=25, random_state=42, verbose=False)
        clf.fit(Ztr, y[trn])
        pv = clf.predict_proba(Zva)[:, 1]
        pt = clf.predict_proba(Zte)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        oof_m[val] = np.clip(iso.fit_transform(pv, y[val]), EPS, 1 - EPS)
        test_m += np.clip(iso.transform(np.clip(pt, 0, 1)), EPS, 1 - EPS) / N_SPLITS
        print(f"  fold {fold}: iters={clf.n_iter_} comb={score_oof(y[val], oof_m[val]):.5f}",
              flush=True)
    sc_mlp = score_oof(y, oof_m)
    print(f"  MLP alone: comb={sc_mlp:.5f}", flush=True)
    best_w, best_sc, best_test = 0.0, score_oof(y, oof_blend), test_blend
    for w in (0.05, 0.10, 0.15, 0.20, 0.30):
        p = np.clip((1 - w) * oof_blend + w * oof_m, EPS, 1 - EPS)
        sc = score_oof(y, p)
        if sc > best_sc + 1e-6:
            best_w, best_sc = w, sc
            best_test = np.clip((1 - w) * test_blend + w * test_m, EPS, 1 - EPS)
    print(f"  MLP blend: best_w={best_w} comb={best_sc:.5f} "
          f"({'GAIN' if best_w else 'no gain'})", flush=True)
    path = None
    if best_w > 0 and best_sc > BAR:
        test_pred = np.clip(best_test, 1e-6, 1 - 1e-6)
        path = os.path.join(OUT, f"{tag}_mlp.csv")
        pd.DataFrame({ID_COL: te_ids, "Target": test_pred}).to_csv(path,
                                                                  index=False)
        print(f"Saved {path} (OOF comb {best_sc:.5f})", flush=True)
    st["stage3_done"] = True
    st["stage3"] = {"score": best_sc, "w": best_w, "path": path}
    save_state(st)
    return best_sc, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="1,2,3",
                    help="comma-separated subset of stages to run")
    args = ap.parse_args()
    stages = set(args.stages.split(","))
    os.makedirs(OUT, exist_ok=True)
    st = load_state()

    tr399, te399, y, feats399, bal_p25 = build_399(seed=42)
    te_ids = te399[ID_COL].values

    kept_v5 = []
    if "1" in stages:
        kept_v5, best1 = stage_greedy_v5(tr399, te399, y, feats399)
    elif st.get("stage1_done"):
        kept_v5 = st["stage1_kept"]
    feats_final = feats399 + kept_v5
    print(f"\nfeats_final={len(feats_final)} (v5 kept: {len(kept_v5)})", flush=True)

    tr0, te0, _ = add_features_v5(tr399.copy(), te399.copy())

    winner_path, winner_score = None, 0.0
    if "2" in stages:
        sc2, path2 = stage_6seed_blend(tr0, te0, y, feats_final, te_ids)
        if path2:
            winner_path, winner_score = path2, sc2
            st["winner"] = {"score": sc2, "file": path2}
            save_state(st)

    if "3" in stages and not winner_path:
        # MLP screen needs the stage-2 blend OOF; recompute cheaply from npz
        oofs_l = [np.load(os.path.join(OUT, f"iter15_lgbm_s{s}.npz"))["oof"]
                  for s in SEEDS6]
        oofs_c = [np.load(os.path.join(OUT, f"iter15_catboost_s{s}.npz"))["oof"]
                  for s in SEEDS6]
        tests_l = [np.load(os.path.join(OUT, f"iter15_lgbm_s{s}.npz"))["test"]
                   for s in SEEDS6]
        tests_c = [np.load(os.path.join(OUT, f"iter15_catboost_s{s}.npz"))["test"]
                   for s in SEEDS6]
        nm, oof_b, test_b, sc_b = blend_bakeoff(y, oofs_l, tests_l, oofs_c,
                                               tests_c, SEEDS6, "iter15-mlpbase")
        sc3, path3 = stage_mlp_screen(y, tr0, te0, feats_final, oof_b, test_b,
                                     te_ids, "iter15")
        if path3:
            winner_path, winner_score = path3, sc3
            st["winner"] = {"score": sc3, "file": path3}
            save_state(st)

    if winner_path:
        print(f"\nWINNER: {winner_path} comb={winner_score:.5f} > bar {BAR}",
              flush=True)
    else:
        print(f"\nNo winner this run (bar {BAR}).", flush=True)
    print("EXIT:0")


if __name__ == "__main__":
    main()
