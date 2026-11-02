"""Iteration 13: CatBoost tune -> FE pass 3 (v3 battery) -> DART/LR stacker -> blend re-tune.

Bar: iter12 OOF combined 0.81556 (6-seed LGBM+CatBoost hybrid A_wg0.5_wl0.5, 319 feats).
Public: us 0.70510 (rank ~87), leader 0.74293. 4 submission slots left today (16/200 used).

Stage 1: CatBoost hyperparam grid (seed 42, 319 feats): depth {6,8} x l2_leaf_reg
  {3,5,8}, then lr {0.02,0.03} x iterations {2000,3000} around the winner.
  Keep the winner only if its seed-42 hybrid with cached LGBM-s42 beats the
  reference hybrid.
Stage 2: v3 feature battery (~70 candidates — friend's v2-style base features
  not yet replicated): per-category monthly slopes, balance trajectory shapes,
  withdrawal ratios with new denominators, activity-rate interactions,
  early/late ratios, concentration, volatility. Greedy batches of 20 on
  seed-42 LGBM, same protocol as iter12 Stage B.
Stage 3: DART booster test (seed 42) and a regularized LR stacker on the
  12-seed OOF matrix. Reject quickly if no gain.
Stage 4: final 6-seed retrain of changed components + hybrid blend re-tune.
  Save submissions/iter13.csv (iter14.csv, ...) for configs beating 0.81556.

RULES: 5-fold stratified CV seed 42, per-fold isotonic. NEVER submits
(parent handles browser). Resume-safe via submissions/iter13_state.json.
Sequential training (4 cores / 7GB RAM machine).
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegressionCV

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iter6
import iter8
import iter12
from iter6 import (add_features_base, add_features_expanded, add_target_encoding,
                   run_lgbm, KITCHEN_SINK, CAT_COLS, N_SPLITS, TARGET, ID_COL,
                   DATA, OUT, EPS, combined)
from iter8 import add_liquidity_features, score_oof, CAT_ORDER
from iter12 import build_279, add_features_v2, hybrid, blend_bakeoff

BAR = 0.81556          # iter12 6-seed hybrid
SEEDS6 = [42, 7, 2025, 5, 11, 777]
BATCH = 20
PATIENCE = 2
MAX_FEATURES = 360
STATE13 = os.path.join(OUT, "iter13_state.json")

CB_REF = {"depth": 6, "learning_rate": 0.03, "iterations": 2000, "l2_leaf_reg": 3.0}


def save_state(st):
    with open(STATE13, "w") as f:
        json.dump(st, f)


def load_state():
    if os.path.exists(STATE13):
        with open(STATE13) as f:
            return json.load(f)
    return {}


def build_319(seed=42):
    """iter12's 319-feature set (279 + 40 kept v2), TE at `seed`."""
    tr, te, y, feats279, bal_p25 = build_279(seed=seed)
    with open(os.path.join(OUT, "iter12_state.json")) as f:
        kept_v2 = json.load(f)["kept_v2"]
    tr2, te2, v2_names = add_features_v2(tr.copy(), te.copy(), bal_p25)
    feats = feats279 + kept_v2
    missing = [c for c in feats if c not in tr2.columns]
    assert not missing, f"missing feats: {missing[:5]}"
    print(f"319-feat set built (seed {seed}): {len(feats)} feats", flush=True)
    return tr2, te2, y, feats, bal_p25


def run_catboost_cfg(X, y, Xte, cfg, tag=""):
    """5-fold CV CatBoost with custom params + per-fold isotonic (seed 42)."""
    iter8.SEED = 42
    oof = np.zeros(len(X))
    test = np.zeros(len(Xte))
    iters = []
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    for fold, (trn, val) in enumerate(skf.split(X, y)):
        Xtr = X.iloc[trn].copy()
        Xva = X.iloc[val].copy()
        Xte_c = Xte.copy()
        for c in CAT_ORDER:
            Xtr[c] = Xtr[c].astype(str)
            Xva[c] = Xva[c].astype(str)
            Xte_c[c] = Xte_c[c].astype(str)
        m = CatBoostClassifier(iterations=cfg["iterations"],
                               learning_rate=cfg["learning_rate"],
                               depth=cfg["depth"],
                               l2_leaf_reg=cfg["l2_leaf_reg"],
                               loss_function="Logloss", eval_metric="Logloss",
                               random_seed=42, early_stopping_rounds=100,
                               use_best_model=True, verbose=False,
                               allow_writing_files=False, thread_count=4)
        m.fit(Xtr, y[trn], eval_set=(Xva, y[val]), cat_features=CAT_ORDER)
        pv = m.predict_proba(Xva)[:, 1]
        pt = m.predict_proba(Xte_c)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        oof[val] = np.clip(iso.fit_transform(pv, y[val]), EPS, 1 - EPS)
        test += np.clip(iso.transform(np.clip(pt, 0, 1)), EPS, 1 - EPS) / N_SPLITS
        iters.append(m.get_best_iteration())
    sc = score_oof(y, oof)
    print(f"  [{tag}] mean_iter={np.mean(iters):.0f} comb={sc:.5f}", flush=True)
    return oof, test


# ---------------------------------------------------------------- Stage 1
def stage1_cb_tune(tr, te, y, feats):
    """CatBoost grid on seed 42. Returns (best_cfg, changed_bool)."""
    print("\n=== Stage 1: CatBoost hyperparam tune (seed 42, 319 feats) ===", flush=True)
    st = load_state()
    if st.get("stage1_done"):
        print("stage1 cached:", st["stage1_summary"], flush=True)
        return st["stage1_summary"]["cfg"], st["stage1_summary"]["changed"]

    X, Xte = tr[feats], te[feats]
    oof_l42 = np.load(os.path.join(OUT, "iter12_lgbm_s42.npz"))["oof"]
    oof_c42_ref = np.load(os.path.join(OUT, "iter12_catboost_s42.npz"))["oof"]
    base_hyb = score_oof(y, hybrid(oof_l42, oof_c42_ref, 0.5, 0.5))
    base_raw = score_oof(y, oof_c42_ref)
    print(f"ref CB s42: raw={base_raw:.5f} hybrid(wg.5/wl.5)={base_hyb:.5f}", flush=True)

    runs = st.get("stage1_runs", {})

    def eval_cfg(cfg):
        tag = (f"cb_d{cfg['depth']}_lr{cfg['learning_rate']}"
               f"_it{cfg['iterations']}_l2{cfg['l2_leaf_reg']}")
        if tag in runs:
            sc_raw, sc_hyb = runs[tag]
            print(f"  [{tag}] cached raw={sc_raw:.5f} hyb={sc_hyb:.5f}", flush=True)
        else:
            oof, _ = run_catboost_cfg(X, y, Xte, cfg, tag)
            sc_raw = score_oof(y, oof)
            sc_hyb = score_oof(y, hybrid(oof_l42, oof, 0.5, 0.5))
            runs[tag] = [sc_raw, sc_hyb]
            st["stage1_runs"] = runs
            save_state(st)
        return tag, runs[tag][0], runs[tag][1]

    # Grid A: depth x l2 (lr=0.03, iter=2000)
    gridA = [{"depth": d, "learning_rate": 0.03, "iterations": 2000, "l2_leaf_reg": l2}
             for d in (6, 8) for l2 in (3, 5, 8)]
    results = [eval_cfg(c) + (c,) for c in gridA]
    best = max(results, key=lambda r: r[2])
    print(f"Grid A winner: {best[0]} raw={best[1]:.5f} hyb={best[2]:.5f}", flush=True)

    # Grid B: lr x iters around the Grid-A winner's depth/l2
    wcfg = best[3]
    gridB = [{"depth": wcfg["depth"], "learning_rate": lr, "iterations": it,
              "l2_leaf_reg": wcfg["l2_leaf_reg"]}
             for lr in (0.02, 0.03) for it in (2000, 3000)]
    seen_cfgs = {(c["depth"], c["learning_rate"], c["iterations"], c["l2_leaf_reg"])
                 for c in gridA}
    for c in gridB:
        key = (c["depth"], c["learning_rate"], c["iterations"], c["l2_leaf_reg"])
        if key in seen_cfgs:
            continue
        seen_cfgs.add(key)
        results.append(eval_cfg(c) + (c,))
    best = max(results, key=lambda r: r[2])
    print(f"Overall winner: {best[0]} raw={best[1]:.5f} hyb={best[2]:.5f}", flush=True)

    changed = best[2] > base_hyb + 1e-6
    summary = {"cfg": best[3], "tag": best[0], "raw": best[1], "hyb": best[2],
               "base_hyb": base_hyb, "changed": changed}
    st["stage1_done"] = True
    st["stage1_summary"] = summary
    save_state(st)
    print(f"Stage 1: winner {'KEPT (beats ref hybrid)' if changed else 'rejected (no gain)'}",
          flush=True)
    return best[3], changed

# ---------------------------------------------------------------- Stage 2
def add_features_v3(tr, te):
    """Third feature battery (~70 candidates, v3_ prefix). Unsupervised."""
    new_tr, new_te = {}, {}

    def add(name, v_tr, v_te):
        if name in tr.columns or name in new_tr:
            raise RuntimeError(f"name collision: {name}")
        new_tr[name] = np.asarray(v_tr, dtype=float)
        new_te[name] = np.asarray(v_te, dtype=float)

    def monthly(df, metric):
        # columns m1..m6, index 0 = m1 (newest)
        return df[[f"m{i}_{metric}" for i in range(1, 7)]].values.astype(float)

    b = monthly(tr, "daily_avg_bal")
    bte = monthly(te, "daily_avg_bal")
    t = np.arange(6, dtype=float)  # 0 = m1 (newest)
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

    def cv(a):
        mu = np.abs(a.mean(axis=1))
        return a.std(axis=1) / np.maximum(mu, 1e-9)

    # 1. per-category monthly slopes (7)
    for c in cats:
        add(f"v3_slope_{c}", slope(tv[c]), slope(tvte[c]))
    # 2. balance trajectory: quadratic a*t^2+b*t (t: 0=newest) + linear slope (3)
    qa = np.polyfit(t, b.T, 2)   # rows: a, b, c
    qate = np.polyfit(t, bte.T, 2)
    add("v3_bal_quad_a", qa[0], qate[0])
    add("v3_bal_quad_b", qa[1], qate[1])
    add("v3_bal_slope", slope(b), slope(bte))
    # 3. withdrawal ratios, new denominators (4)
    add("v3_wd1_div_outflow1", tv["withdraw"][:, 0] / np.maximum(outflow[:, 0], 1.0),
        tvte["withdraw"][:, 0] / np.maximum(outflowte[:, 0], 1.0))
    add("v3_wd6_div_inflow6", tv["withdraw"].sum(1) / np.maximum(inflow.sum(1), 1.0),
        tvte["withdraw"].sum(1) / np.maximum(inflowte.sum(1), 1.0))
    add("v3_wd6_div_dp6", tv["withdraw"].sum(1) / np.maximum(tv["deposit"].sum(1), 1.0),
        tvte["withdraw"].sum(1) / np.maximum(tvte["deposit"].sum(1), 1.0))
    add("v3_wd1_div_dp1", tv["withdraw"][:, 0] / np.maximum(tv["deposit"][:, 0], 1.0),
        tvte["withdraw"][:, 0] / np.maximum(tvte["deposit"][:, 0], 1.0))
    # 4. activity-rate interactions (4)
    inflow6 = inflow.sum(1)
    inflow6te = inflowte.sum(1)
    add("v3_act_x_inflow6", act * inflow6 / np.maximum(balmean, 1.0),
        actte * inflow6te / np.maximum(balmeante, 1.0))
    add("v3_act_x_outflow6", act * outflow.sum(1) / np.maximum(balmean, 1.0),
        actte * outflowte.sum(1) / np.maximum(balmeante, 1.0))
    add("v3_act_x_netflow6", act * net.sum(1) / np.maximum(balmean, 1.0),
        actte * nette.sum(1) / np.maximum(balmeante, 1.0))
    add("v3_act_x_wd1", act * tv["withdraw"][:, 0],
        actte * tvte["withdraw"][:, 0])
    # 5. early-vs-late ratios (m4+m5+m6 oldest over m1+m2+m3 newest) (4)
    for c, arr, arrte in (("wd", tv["withdraw"], tvte["withdraw"]),
                          ("dp", tv["deposit"], tvte["deposit"]),
                          ("inflow", inflow, inflowte),
                          ("outflow", outflow, outflowte)):
        add(f"v3_earlylate_{c}",
            arr[:, 3:].sum(1) / np.maximum(arr[:, :3].sum(1), 1.0),
            arrte[:, 3:].sum(1) / np.maximum(arrte[:, :3].sum(1), 1.0))
    # 6. formal-vs-cash (2)
    formal = tv["paybill"] + tv["merchantpay"]
    formalte = tvte["paybill"] + tvte["merchantpay"]
    add("v3_formal_div_cash_m1", formal[:, 0] / np.maximum(tv["withdraw"][:, 0], 1.0),
        formalte[:, 0] / np.maximum(tvte["withdraw"][:, 0], 1.0))
    add("v3_formal_div_cash_6m", formal.sum(1) / np.maximum(tv["withdraw"].sum(1), 1.0),
        formalte.sum(1) / np.maximum(tvte["withdraw"].sum(1), 1.0))
    # 7. mm_send / received (2)
    add("v3_mmsend_div_recv_m1",
        tv["mm_send"][:, 0] / np.maximum(tv["received"][:, 0], 1.0),
        tvte["mm_send"][:, 0] / np.maximum(tvte["received"][:, 0], 1.0))
    add("v3_mmsend_div_recv_6m",
        tv["mm_send"].sum(1) / np.maximum(tv["received"].sum(1), 1.0),
        tvte["mm_send"].sum(1) / np.maximum(tvte["received"].sum(1), 1.0))
    # 8. volatility: CV per category (4)
    for c, arr, arrte in (("wd", tv["withdraw"], tvte["withdraw"]),
                          ("dp", tv["deposit"], tvte["deposit"]),
                          ("inflow", inflow, inflowte),
                          ("outflow", outflow, outflowte)):
        add(f"v3_cv_{c}", cv(arr), cv(arrte))
    # 9. argmax/argmin months (4)
    add("v3_argmax_bal", np.argmax(b, axis=1).astype(float),
        np.argmax(bte, axis=1).astype(float))
    add("v3_argmin_bal", np.argmin(b, axis=1).astype(float),
        np.argmin(bte, axis=1).astype(float))
    add("v3_argmax_wd", np.argmax(tv["withdraw"], axis=1).astype(float),
        np.argmax(tvte["withdraw"], axis=1).astype(float))
    add("v3_argmax_inflow", np.argmax(inflow, axis=1).astype(float),
        np.argmax(inflowte, axis=1).astype(float))
    # 10. dp/wd dynamics (2)
    r = tv["deposit"] / np.maximum(tv["withdraw"], 1.0)
    rte = tvte["deposit"] / np.maximum(tvte["withdraw"], 1.0)
    add("v3_dp_div_wd_m1", r[:, 0], rte[:, 0])
    add("v3_dp_div_wd_slope", slope(r), slope(rte))
    # 11. received/deposit split (1)
    add("v3_recv_div_dp_mean",
        (tv["received"] / np.maximum(tv["deposit"], 1.0)).mean(axis=1),
        (tvte["received"] / np.maximum(tvte["deposit"], 1.0)).mean(axis=1))
    # 12. balance coverage (2)
    add("v3_bal1_minus_outflow1", b[:, 0] - outflow[:, 0],
        bte[:, 0] - outflowte[:, 0])
    add("v3_bal1_minus_outflow6avg", b[:, 0] - outflow.sum(1) / 6.0,
        bte[:, 0] - outflowte.sum(1) / 6.0)
    # 13. netflow sign flips (1)
    add("v3_netflow_signflips", (np.diff(np.sign(net), axis=1) != 0).sum(axis=1).astype(float),
        (np.diff(np.sign(nette), axis=1) != 0).sum(axis=1).astype(float))
    # 14. max cumulative drawdown of netflow (1)
    cum = np.cumsum(net[:, ::-1], axis=1)  # oldest -> newest
    cumte = np.cumsum(nette[:, ::-1], axis=1)
    dd = np.maximum.accumulate(cum, axis=1) - cum
    ddte = np.maximum.accumulate(cumte, axis=1) - cumte
    add("v3_netflow_maxdd", dd.max(axis=1), ddte.max(axis=1))
    # 15. savings rate (2)
    sr = (inflow - outflow) / np.maximum(inflow, 1.0)
    srte = (inflowte - outflowte) / np.maximum(inflowte, 1.0)
    add("v3_savings_mean", sr.mean(axis=1), srte.mean(axis=1))
    add("v3_savings_min", sr.min(axis=1), srte.min(axis=1))
    # 16. recent withdrawal acceleration (1)
    add("v3_wd_accel_recent",
        (tv["withdraw"][:, 0] - tv["withdraw"][:, 1])
        - (tv["withdraw"][:, 1] - tv["withdraw"][:, 2]),
        (tvte["withdraw"][:, 0] - tvte["withdraw"][:, 1])
        - (tvte["withdraw"][:, 1] - tvte["withdraw"][:, 2]))
    # 17. balance drop variants (2)
    bmax = b.max(axis=1)
    bmaxte = bte.max(axis=1)
    add("v3_bal_drop_maxmin", (bmax - b.min(axis=1)) / np.maximum(bmax, 1.0),
        (bmaxte - bte.min(axis=1)) / np.maximum(bmaxte, 1.0))
    add("v3_bal_drop_max_m1", (bmax - b[:, 0]) / np.maximum(bmax, 1.0),
        (bmaxte - bte[:, 0]) / np.maximum(bmaxte, 1.0))
    # 18. concentration (2)
    add("v3_inflow_concentration", inflow.max(axis=1) / np.maximum(inflow6, 1.0),
        inflowte.max(axis=1) / np.maximum(inflow6te, 1.0))
    add("v3_outflow_concentration", outflow.max(axis=1) / np.maximum(outflow.sum(1), 1.0),
        outflowte.max(axis=1) / np.maximum(outflowte.sum(1), 1.0))
    # 19. deposit regularity (1)
    add("v3_dp_regularity", 1.0 / (1.0 + cv(tv["deposit"])),
        1.0 / (1.0 + cv(tvte["deposit"])))
    # 20. age interactions (2)
    add("v3_age_x_bal_slope", age * slope(b), agete * slope(bte))
    add("v3_age_x_netflow6", age * net.sum(1) / np.maximum(balmean, 1.0),
        agete * nette.sum(1) / np.maximum(balmeante, 1.0))
    # 21. avg ticket size (3)
    add("v3_ticket_pb_m1",
        tv["paybill"][:, 0] / np.maximum(vol["paybill"][:, 0], 1.0),
        tvte["paybill"][:, 0] / np.maximum(volte["paybill"][:, 0], 1.0))
    add("v3_ticket_wd_m1",
        tv["withdraw"][:, 0] / np.maximum(vol["withdraw"][:, 0], 1.0),
        tvte["withdraw"][:, 0] / np.maximum(volte["withdraw"][:, 0], 1.0))
    tick_wd = tv["withdraw"] / np.maximum(vol["withdraw"], 1.0)
    tick_wdte = tvte["withdraw"] / np.maximum(volte["withdraw"], 1.0)
    add("v3_ticket_wd_slope", slope(tick_wd), slope(tick_wdte))
    # 22. transfer_from_bank ratios (2)
    add("v3_tfb_div_inflow6", tv["transfer_from_bank"].sum(1) / np.maximum(inflow6, 1.0),
        tvte["transfer_from_bank"].sum(1) / np.maximum(inflow6te, 1.0))
    add("v3_tfb_div_outflow6",
        tv["transfer_from_bank"].sum(1) / np.maximum(outflow.sum(1), 1.0),
        tvte["transfer_from_bank"].sum(1) / np.maximum(outflowte.sum(1), 1.0))
    # 23. misc trajectory (6)
    add("v3_outflow_slope", slope(outflow), slope(outflowte))
    add("v3_netflow_slope", slope(net), slope(nette))
    add("v3_bal_velocity", (b[:, 0] - b[:, 5]) / 6.0, (bte[:, 0] - bte[:, 5]) / 6.0)
    add("v3_wd_share_outflow_mean",
        (tv["withdraw"] / np.maximum(outflow, 1.0)).mean(axis=1),
        (tvte["withdraw"] / np.maximum(outflowte, 1.0)).mean(axis=1))
    add("v3_dp_share_inflow_mean",
        (tv["deposit"] / np.maximum(inflow, 1.0)).mean(axis=1),
        (tvte["deposit"] / np.maximum(inflowte, 1.0)).mean(axis=1))
    add("v3_outflow_div_arpu", outflow[:, 0] / np.maximum(arpu, 1.0),
        outflowte[:, 0] / np.maximum(arpute, 1.0))

    for name in new_tr:
        tr[name] = new_tr[name]
        te[name] = new_te[name]
    names = list(new_tr.keys())
    print(f"v3 battery: {len(names)} candidates", flush=True)
    return tr, te, names


def stage2_fe_pass3(tr, te, y, feats319):
    """Greedy forward selection of v3 candidates in gain-ranked batches of 20."""
    print("\n=== Stage 2: FE pass 3 (v3 battery) ===", flush=True)
    st = load_state()
    if st.get("stage2_done"):
        print("stage2 cached: kept", len(st["stage2_kept"]), "best", st["stage2_best"],
              flush=True)
        return st["stage2_kept"], st["stage2_best"]

    iter6.SEED = 42
    tr3, te3, v3_names = add_features_v3(tr.copy(), te.copy())

    # gain ranking from a full-data fit
    if "ranked_v3" not in st:
        print("gain-ranking fit on full data (319+v3)...", flush=True)
        Xg = tr3[feats319 + v3_names]
        dall = lgb.Dataset(Xg, y, categorical_feature=CAT_COLS)
        m = lgb.train(dict(KITCHEN_SINK, num_threads=4), dall, num_boost_round=800)
        gain = pd.Series(m.feature_importance(importance_type="gain"),
                         index=feats319 + v3_names)
        st["ranked_v3"] = [f for f in gain.sort_values(ascending=False).index
                           if f in v3_names]
        save_state(st)
    ranked_v3 = st["ranked_v3"]

    kept = st.get("stage2_kept", [])
    done_batches = set(st.get("stage2_batches", []))
    # baseline: 319-feat LGBM seed-42 (recompute for identical protocol)
    if "stage2_best" not in st:
        oof0, _, _ = run_lgbm(tr3[feats319], y, te3[feats319], KITCHEN_SINK)
        st["stage2_best"] = score_oof(y, oof0)
        print(f"stage2 baseline (319, seed42): {st['stage2_best']:.5f}", flush=True)
        save_state(st)
    best = st["stage2_best"]

    batches = [ranked_v3[i:i + BATCH] for i in range(0, len(ranked_v3), BATCH)]
    no_gain = st.get("stage2_nogain", 0)
    for bi, batch in enumerate(batches):
        if bi in done_batches:
            print(f"batch {bi}: skipped (done)", flush=True)
            continue
        if len(feats319) + len(kept) + len(batch) > MAX_FEATURES:
            print(f"batch {bi}: skipped (would exceed {MAX_FEATURES})", flush=True)
            done_batches.add(bi)
            continue
        cand = feats319 + kept + batch
        oof, _, _ = run_lgbm(tr3[cand], y, te3[cand], KITCHEN_SINK)
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
    print(f"Stage 2 done: kept {len(kept)} v3 features, best={best:.5f}", flush=True)
    return kept, best

# ---------------------------------------------------------------- Stage 3
def run_lgbm_dart(X, y, Xte, params, num_round=800):
    """DART has no early stopping: fixed rounds + per-fold isotonic."""
    oof = np.zeros(len(X))
    test = np.zeros(len(Xte))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    for trn, val in skf.split(X, y):
        dtr = lgb.Dataset(X.iloc[trn], y[trn], categorical_feature=CAT_COLS)
        m = lgb.train(params, dtr, num_boost_round=num_round)
        pv = m.predict(X.iloc[val])
        pt = m.predict(Xte)
        iso = IsotonicRegression(out_of_bounds="clip")
        oof[val] = np.clip(iso.fit_transform(pv, y[val]), EPS, 1 - EPS)
        test += np.clip(iso.transform(np.clip(pt, 0, 1)), EPS, 1 - EPS) / N_SPLITS
    return oof, test


def stage3_dart_lr(tr, te, y, feats):
    """DART booster test (seed 42) + regularized LR stacker on 12-seed OOF matrix.
    Returns dict with results."""
    print("\n=== Stage 3: DART test + LR stacker ===", flush=True)
    st = load_state()
    if st.get("stage3_done"):
        print("stage3 cached:", st["stage3"], flush=True)
        return st["stage3"]
    res = {}

    # 3a. DART vs gbdt at seed 42, same feats (DART: fixed 800 rounds)
    iter6.SEED = 42
    print("training gbdt reference (seed 42, final feats)...", flush=True)
    oof_gbdt, _, _ = run_lgbm(tr[feats], y, te[feats], KITCHEN_SINK)
    ref_lgbm = score_oof(y, oof_gbdt)
    dart_params = dict(KITCHEN_SINK)
    dart_params.update(boosting_type="dart", drop_rate=0.15, max_drop=50,
                       skip_drop=0.5)
    print("training DART (seed 42, 800 rounds)...", flush=True)
    oof_dart, _ = run_lgbm_dart(tr[feats], y, te[feats], dart_params)
    sc_dart = score_oof(y, oof_dart)
    res["dart"] = sc_dart
    res["lgbm_gbdt_s42"] = ref_lgbm
    res["dart_wins"] = bool(sc_dart > ref_lgbm + 1e-6)
    print(f"DART s42={sc_dart:.5f} vs gbdt s42={ref_lgbm:.5f} "
          f"({'DART WINS' if res['dart_wins'] else 'gbdt holds'})", flush=True)

    # 3b. LR stacker on the 12 seed-OOF columns (iter12 caches, 319 feats)
    meta_oof = np.column_stack(
        [np.load(os.path.join(OUT, f"iter12_lgbm_s{s}.npz"))["oof"] for s in SEEDS6]
        + [np.load(os.path.join(OUT, f"iter12_catboost_s{s}.npz"))["oof"] for s in SEEDS6])
    lr = LogisticRegressionCV(Cs=8, cv=5, max_iter=5000, n_jobs=4)
    meta_pred = cross_val_predict(lr, meta_oof, y, cv=5, method="predict_proba")[:, 1]
    sc_lr = score_oof(y, np.clip(meta_pred, EPS, 1 - EPS))
    sc_mean12 = score_oof(y, np.clip(meta_oof.mean(axis=1), EPS, 1 - EPS))
    res["lr_stacker"] = sc_lr
    res["mean12"] = sc_mean12
    res["lr_wins_blend"] = bool(sc_lr > BAR + 1e-6)
    print(f"LR stacker honest-OOF={sc_lr:.5f} vs plain-mean12={sc_mean12:.5f} "
          f"(bar {BAR}; {'BEATS BAR' if res['lr_wins_blend'] else 'below bar'})", flush=True)

    st["stage3_done"] = True
    st["stage3"] = res
    save_state(st)
    return res


# ---------------------------------------------------------------- Stage 4
def train_dart_seed13(seed, tr0, te0, y, feats, tag_prefix="iter13"):
    """6-seed DART (fixed 800 rounds). Resume-safe npz."""
    iter6.SEED = seed
    p = os.path.join(OUT, f"{tag_prefix}_dart_s{seed}.npz")
    if os.path.exists(p):
        d = np.load(p)
        print(f"seed {seed}: DART loaded cached", flush=True)
        return d["oof"], d["test"]
    tr = tr0.copy()
    te = te0.copy()
    tr, te, _ = add_target_encoding(tr, te, y)
    prm = dict(KITCHEN_SINK)
    prm.update(boosting_type="dart", drop_rate=0.15, max_drop=50, skip_drop=0.5)
    print(f"seed {seed}: training DART...", flush=True)
    oof, test = run_lgbm_dart(tr[feats], y, te[feats], prm)
    np.savez(p, oof=oof, test=test)
    print(f"seed {seed}: DART done, comb={score_oof(y, oof):.5f}", flush=True)
    return oof, test


def train_lgbm_seed13(seed, tr0, te0, y, feats, params, tag_prefix="iter13"):
    iter6.SEED = seed
    p = os.path.join(OUT, f"{tag_prefix}_lgbm_s{seed}.npz")
    if os.path.exists(p):
        d = np.load(p)
        print(f"seed {seed}: LGBM loaded cached", flush=True)
        return d["oof"], d["test"]
    tr = tr0.copy()
    te = te0.copy()
    tr, te, _ = add_target_encoding(tr, te, y)
    prm = dict(params, seed=seed)
    print(f"seed {seed}: training LGBM...", flush=True)
    oof, test, mi = run_lgbm(tr[feats], y, te[feats], prm)
    np.savez(p, oof=oof, test=test, iters=mi)
    print(f"seed {seed}: LGBM done, mean_iter={mi:.0f}, comb={score_oof(y, oof):.5f}",
          flush=True)
    return oof, test


def train_cb_seed13(seed, tr0, te0, y, feats, cb_cfg, tag_prefix="iter13"):
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
            Xte_c[c] = Xte_c[c].astype(str)
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
        f.write(f"ITERATION 13 ({tag}) results\nbar (iter12 6-seed): {BAR}\n")
        f.write(f"FINAL OOF combined={score:.5f} BEATS {BAR}\n")
        f.write(f"spec: {json.dumps(spec)}\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="1,2,3,4",
                    help="comma-separated subset of stages to run")
    args = ap.parse_args()
    stages = set(args.stages.split(","))
    os.makedirs(OUT, exist_ok=True)
    st = load_state()

    # ---- base 319 set (seed 42) ----
    tr319, te319, y, feats319, bal_p25 = build_319(seed=42)
    te_ids = te319[ID_COL].values

    # ---- Stage 1: CatBoost tune ----
    cb_cfg, cb_changed = (CB_REF, False)
    if "1" in stages:
        cb_cfg, cb_changed = stage1_cb_tune(tr319, te319, y, feats319)
    elif st.get("stage1_done"):
        s = st["stage1_summary"]
        cb_cfg, cb_changed = s["cfg"], s["changed"]

    # ---- Stage 2: FE pass 3 ----
    kept_v3 = []
    if "2" in stages:
        kept_v3, best2 = stage2_fe_pass3(tr319, te319, y, feats319)
    elif st.get("stage2_done"):
        kept_v3 = st["stage2_kept"]
    feats_final = feats319 + kept_v3
    feats_changed = len(kept_v3) > 0
    print(f"\nfeats_final={len(feats_final)} (v3 kept: {len(kept_v3)}), "
          f"cb_changed={cb_changed}", flush=True)

    # tr0/te0 carry v3 columns (unsupervised); TE re-done per seed in trainers
    if feats_changed:
        tr0, te0, _ = add_features_v3(tr319.copy(), te319.copy())
    else:
        tr0, te0 = tr319, te319

    # ---- Stage 3: DART + LR stacker ----
    s3 = {"dart_wins": False, "lr_wins_blend": False}
    if "3" in stages:
        s3 = stage3_dart_lr(tr0, te0, y, feats_final)
    elif st.get("stage3_done"):
        s3 = st["stage3"]

    dart_wins = bool(s3.get("dart_wins"))
    if dart_wins:
        print("DART won screening -> using DART for the 6-seed LGBM slot", flush=True)

    # ---- Stage 4: final 6-seed assembly + blend re-tune ----
    saved = []
    if "4" in stages:
        print("\n=== Stage 4: final 6-seed assembly + blend re-tune ===", flush=True)

        need_lgbm = feats_changed or dart_wins
        need_cb = feats_changed or cb_changed
        tag_prefix = "iter13"
        oofs_l, tests_l, oofs_c, tests_c = [], [], [], []
        for s in SEEDS6:
            if need_lgbm:
                if dart_wins:
                    oof_l, test_l = train_dart_seed13(s, tr0, te0, y, feats_final,
                                                     tag_prefix)
                else:
                    oof_l, test_l = train_lgbm_seed13(s, tr0, te0, y, feats_final,
                                                     KITCHEN_SINK, tag_prefix)
            else:
                d = np.load(os.path.join(OUT, f"iter12_lgbm_s{s}.npz"))
                oof_l, test_l = d["oof"], d["test"]
                print(f"seed {s}: LGBM reused iter12 cache", flush=True)
            oofs_l.append(oof_l)
            tests_l.append(test_l)
            if need_cb:
                oof_c, test_c = train_cb_seed13(s, tr0, te0, y, feats_final,
                                               cb_cfg, tag_prefix)
            else:
                d = np.load(os.path.join(OUT, f"iter12_catboost_s{s}.npz"))
                oof_c, test_c = d["oof"], d["test"]
                print(f"seed {s}: CatBoost reused iter12 cache", flush=True)
            oofs_c.append(oof_c)
            tests_c.append(test_c)

        nm, oof_b, test_b, sc_b = blend_bakeoff(y, oofs_l, tests_l, oofs_c,
                                               tests_c, SEEDS6, "iter13-6seed")
        best_nm, best_oof, best_test, best_sc = nm, oof_b, test_b, sc_b
        best_kind = f"hybrid:{nm}"

        # LR stacker on the FINAL 12 seed-OOF columns (honest CV)
        meta_oof = np.column_stack(oofs_l + oofs_c)
        lr = LogisticRegressionCV(Cs=8, cv=5, max_iter=5000, n_jobs=4)
        meta_pred = cross_val_predict(lr, meta_oof, y, cv=5,
                                      method="predict_proba")[:, 1]
        sc_lr = score_oof(y, np.clip(meta_pred, EPS, 1 - EPS))
        print(f"LR stacker (final members) honest-OOF={sc_lr:.5f} "
              f"vs hybrid={sc_b:.5f}", flush=True)
        if sc_lr > sc_b + 1e-6:
            lr.fit(meta_oof, y)
            meta_test = np.column_stack(tests_l + tests_c)
            best_test = np.clip(lr.predict_proba(meta_test)[:, 1], EPS, 1 - EPS)
            best_oof = np.clip(meta_pred, EPS, 1 - EPS)
            best_sc = sc_lr
            best_kind = "lr_stacker"
            print("LR stacker WINS the final blend", flush=True)

        print(f"\nStage 4 final: {best_kind} OOF comb={best_sc:.5f} "
              f"({'BEATS' if best_sc > BAR else 'below'} bar {BAR})", flush=True)
        if best_sc > BAR:
            spec = {"kind": best_kind, "blend": nm, "seeds": SEEDS6,
                    "feats": len(feats_final), "v3_kept": kept_v3,
                    "cb_cfg": cb_cfg, "cb_changed": cb_changed,
                    "dart": dart_wins,
                    "lgbm": "dart800" if dart_wins else "kitchen_sink"}
            # iter13.csv for the first winner; iter14.csv if a distinct second
            # winner also beats the bar (checked below via stage deltas)
            p = save_submission_csv(best_test, te_ids, "iter13", best_sc, spec)
            saved.append(p)
            st["final"] = {"score": best_sc, "spec": spec, "file": p}
            save_state(st)
        else:
            print("No candidate beat the bar; no CSV written.", flush=True)

    print("EXIT:0")


if __name__ == "__main__":
    main()
