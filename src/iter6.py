"""Iteration 6: feature engineering expansion (research rank #1, highest ROI).

Context: iter5 kitchen_sink (num_leaves=31, l2=3.0, l1=0.5, lr=0.03) hit
CV combined 0.80557, but CV gains are NOT translating to public LB
(iter4 public 0.70008 > iter5 public 0.69907). We need genuinely new signal,
not more tuning. Per the research report section 3, expand add_features()
systematically per (metric, stat) pairs, preferring ratios/rates over
absolute volumes.

Pipeline kept identical for comparability:
- kitchen_sink LGBM params, 5-fold stratified CV seed 42,
  per-fold isotonic calibration, combined = 0.6*(1-LL) + 0.4*AUC.
- Base = iter5's 40 engineered features (add_features_base, verbatim).
- New battery (~213 features): EWMA recency, trend shape (slope3, streaks,
  sign changes), volatility (CV, range, drawdown), extremes (argmin/argmax),
  cross-category shares, per-month net/cumulative/runway, behavioral
  (zero-activity months, low-balance months), static interactions,
  5-fold OOF target encoding for the 5 categoricals.
- Greedy forward selection in gain-ranked batches of 25 on CV combined,
  cap ~300 total features, early-stop after 3 non-improving batches.

RESUME-SAFE: full-model gain run + each greedy batch eval is logged to
submissions/iter6_search.csv; kept-feature state persisted to
submissions/iter6_state.json after every batch. Restart skips done work.

Winner must beat iter5's 0.80557 on the same seed-42 scheme.
Output: submissions/iter6.csv, oof_iter6.csv (NOT submitted here —
daily limit reached; file left ready for tomorrow's reset).
"""
import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression

SEED = 42
N_SPLITS = 5
TARGET = "liquidity_stress_next_30d"
ID_COL = "ID"
CAT_COLS = ["gender", "region", "smartphone", "segment", "earning_pattern"]
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "submissions")
ITER5_BAR = 0.80557  # must beat this on the same seed-42 5-fold scheme
MAX_FEATURES = 300
BATCH = 25
PATIENCE = 3
EPS = 1e-6

BAL_COLS = [f"m{i}_daily_avg_bal" for i in range(6, 0, -1)]  # m6 oldest .. m1 newest
GROUPS = ["deposit", "withdraw", "received", "mm_send", "paybill",
          "merchantpay", "transfer_from_bank"]

LGBM_BASE = {
    "objective": "binary", "metric": "binary_logloss",
    "learning_rate": 0.04, "num_leaves": 63, "min_child_samples": 40,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "verbose": -1, "num_threads": 4, "seed": SEED,
}
KITCHEN_SINK = dict(LGBM_BASE, num_leaves=31, lambda_l2=3.0, lambda_l1=0.5,
                    learning_rate=0.03)


def add_features_base(df):
    """Verbatim from iter5: the 40 base engineered features."""
    df = df.copy()
    b = df[BAL_COLS].values.astype(float)
    t = np.arange(6)  # 0=m6(oldest) .. 5=m1(newest)
    tc = t - t.mean()
    df["bal_slope"] = (b * tc).sum(axis=1) / (tc ** 2).sum()
    df["bal_std"] = b.std(axis=1)
    df["bal_min"] = b.min(axis=1)
    df["bal_max"] = b.max(axis=1)
    df["bal_mean"] = b.mean(axis=1)
    m1 = df["m1_daily_avg_bal"].values
    m6 = df["m6_daily_avg_bal"].values
    df["bal_m1_div_m6"] = m1 / np.maximum(m6, 1.0)
    df["bal_m1_minus_m6"] = m1 - m6
    mom = np.diff(b[:, ::-1], axis=1)  # m1-m2, m2-m3, ... (newest first diffs)
    df["bal_max_drop"] = (-mom).max(axis=1)
    df["bal_n_drops"] = (mom < 0).sum(axis=1)

    for g in GROUPS:
        tv = df[[f"m{i}_{g}_total_value" for i in range(1, 7)]].values.astype(float)
        recent = tv[:, 0] + tv[:, 1]
        old = tv[:, 4] + tv[:, 5]
        df[f"{g}_tv_sum"] = tv.sum(axis=1)
        df[f"{g}_tv_recent_div_old"] = recent / np.maximum(old, 1.0)
        df[f"{g}_tv_recent_minus_old"] = recent - old

    flows = []
    for i in range(1, 7):
        inflow = df[f"m{i}_deposit_total_value"] + df[f"m{i}_received_total_value"]
        outflow = df[f"m{i}_withdraw_total_value"] + df[f"m{i}_mm_send_total_value"]
        flows.append((inflow - outflow).values)
    f = np.stack(flows, axis=1).astype(float)  # m1..m6
    fr = f[:, ::-1]  # m6..m1 oldest first
    df["netflow_sum"] = fr.sum(axis=1)
    df["netflow_slope"] = (fr * tc).sum(axis=1) / (tc ** 2).sum()
    df["netflow_m1"] = f[:, 0]
    wsum = df["withdraw_tv_sum"].values
    dsum = df["deposit_tv_sum"].values
    df["withdraw_div_deposit"] = wsum / np.maximum(dsum, 1.0)
    return df


def _longest_run(b):
    """Longest consecutive run of 1s per row of binary matrix b (n,k)."""
    n, k = b.shape
    best = np.zeros(n, dtype=int)
    cur = np.zeros(n, dtype=int)
    for j in range(k):
        cur = np.where(b[:, j] == 1, cur + 1, 0)
        best = np.maximum(best, cur)
    return best


def _series_stats(prefix, vals):
    """Full stat battery. vals: (n,6) newest-first (m1..m6)."""
    d = {}
    for alpha, aname in ((0.5, "ewma05"), (0.7, "ewma07")):
        w = np.array([alpha * (1 - alpha) ** i for i in range(6)])
        w = w / w.sum()
        d[f"{prefix}_{aname}"] = (vals * w).sum(axis=1)
    rest_mean = vals[:, 1:].mean(axis=1)
    d[f"{prefix}_m1_div_rest"] = vals[:, 0] / np.maximum(np.abs(rest_mean), 1.0)
    d[f"{prefix}_m1_m2_diff"] = vals[:, 0] - vals[:, 1]
    d[f"{prefix}_m1_m3_diff"] = vals[:, 0] - vals[:, 2]
    t3 = np.arange(3)
    tc3 = t3 - t3.mean()
    last3 = vals[:, :3][:, ::-1]  # oldest..newest within last 3 months
    d[f"{prefix}_slope3"] = (last3 * tc3).sum(axis=1) / (tc3 ** 2).sum()
    diffs = np.diff(vals, axis=1)  # m1-m2 .. m5-m6
    d[f"{prefix}_sign_changes"] = ((diffs[:, :-1] * diffs[:, 1:]) < 0).sum(axis=1)
    d[f"{prefix}_n_increasing"] = (diffs > 0).sum(axis=1)
    d[f"{prefix}_longest_inc"] = _longest_run((diffs > 0).astype(int))
    d[f"{prefix}_longest_dec"] = _longest_run((diffs < 0).astype(int))
    d[f"{prefix}_m1_minus_m6"] = vals[:, 0] - vals[:, 5]
    mu = vals.mean(axis=1)
    sd = vals.std(axis=1)
    d[f"{prefix}_cv"] = sd / np.maximum(np.abs(mu), 1.0)
    rng = vals.max(axis=1) - vals.min(axis=1)
    d[f"{prefix}_range"] = rng
    d[f"{prefix}_range_div_mean"] = rng / np.maximum(np.abs(mu), 1.0)
    d[f"{prefix}_argmin_m"] = vals.argmin(axis=1) + 1  # 1=m1 (newest)
    d[f"{prefix}_argmax_m"] = vals.argmax(axis=1) + 1
    d[f"{prefix}_min_div_mean"] = vals.min(axis=1) / np.maximum(np.abs(mu), 1.0)
    d[f"{prefix}_max_div_mean"] = vals.max(axis=1) / np.maximum(np.abs(mu), 1.0)
    return d


def _light_stats(prefix, vals):
    """Cheap battery for noisier series (volumes)."""
    d = {}
    w = np.array([0.5 * 0.5 ** i for i in range(6)])
    w = w / w.sum()
    d[f"{prefix}_ewma05"] = (vals * w).sum(axis=1)
    rest_mean = vals[:, 1:].mean(axis=1)
    d[f"{prefix}_m1_div_rest"] = vals[:, 0] / np.maximum(np.abs(rest_mean), 1.0)
    t3 = np.arange(3)
    tc3 = t3 - t3.mean()
    last3 = vals[:, :3][:, ::-1]
    d[f"{prefix}_slope3"] = (last3 * tc3).sum(axis=1) / (tc3 ** 2).sum()
    mu = vals.mean(axis=1)
    d[f"{prefix}_cv"] = vals.std(axis=1) / np.maximum(np.abs(mu), 1.0)
    d[f"{prefix}_range"] = vals.max(axis=1) - vals.min(axis=1)
    return d


def add_features_expanded(df, bal_p25):
    """New feature battery (~213 features). df already has base features."""
    df = df.copy()
    new = {}

    def monthly(metric):
        # (n,6) newest-first: m1..m6
        return df[[f"m{i}_{metric}" for i in range(1, 7)]].values.astype(float)

    # 1-4. Heavy battery: total_value per group + balance (8 series x 18)
    for g in GROUPS:
        new.update(_series_stats(f"{g}_tv", monthly(f"{g}_total_value")))
    new.update(_series_stats("bal", monthly("daily_avg_bal")))

    # Light battery: volume per group (7 x 5)
    for g in GROUPS:
        new.update(_light_stats(f"{g}_vol", monthly(f"{g}_volume")))

    bal = monthly("daily_avg_bal")
    # max drawdown: strongest liquidity-stress signal per research
    new["bal_max_drawdown"] = bal.max(axis=1) - bal[:, 0]
    new["bal_n_below_p25"] = (bal < bal_p25).sum(axis=1)

    # 5. Cross-category structure
    tv_6m = {g: monthly(f"{g}_total_value").sum(axis=1) for g in GROUPS}
    tv_tot_6m = sum(tv_6m.values())
    for g in GROUPS:
        new[f"{g}_share_6m"] = tv_6m[g] / np.maximum(tv_tot_6m, 1.0)
        m1g = monthly(f"{g}_total_value")[:, 0]
        m1tot = sum(monthly(f"{gg}_total_value")[:, 0] for gg in GROUPS)
        new[f"{g}_share_m1"] = m1g / np.maximum(m1tot, 1.0)
    # per-month net (inflow-outflow) extras beyond base
    nets = []
    outflows = []
    for i in range(1, 7):
        inflow = (df[f"m{i}_deposit_total_value"] + df[f"m{i}_received_total_value"]).values
        outflow = (df[f"m{i}_withdraw_total_value"] + df[f"m{i}_mm_send_total_value"]
                   + df[f"m{i}_paybill_total_value"] + df[f"m{i}_merchantpay_total_value"]).values
        nets.append(inflow - outflow)
        outflows.append(outflow)
    nets = np.stack(nets, axis=1).astype(float)      # m1..m6
    outflows = np.stack(outflows, axis=1).astype(float)
    new["netflow_min"] = nets.min(axis=1)
    new["netflow_last3"] = nets[:, :3].sum(axis=1)
    new["months_of_runway"] = bal[:, 0] / np.maximum(outflows.mean(axis=1), 1.0)

    # 6. Behavioral
    for g in GROUPS:
        vol = monthly(f"{g}_volume")
        new[f"{g}_zero_months"] = (vol == 0).sum(axis=1)

    # Static interactions
    new["arpu_x_bal_change"] = df["arpu"].values * df["bal_m1_minus_m6"].values
    new["activity_x_bal_std"] = df["x_90_d_activity_rate"].values * df["bal_std"].values
    new["age_x_bal_slope"] = df["age"].values * df["bal_slope"].values

    for k, v in new.items():
        df[k] = v
    return df, list(new.keys())


def add_target_encoding(tr, te, y):
    """5-fold OOF smoothed target encoding for the 5 categoricals (train);
    full-train smoothed means for test. Raw categoricals are kept too."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    gm = float(y.mean())
    new_cols = []
    for c in CAT_COLS:
        oof = np.full(len(tr), gm)
        for trn, val in skf.split(tr, y):
            ktr = tr[c].iloc[trn].astype(str).values
            kval = tr[c].iloc[val].astype(str).values
            yt = y[trn]
            df_tmp = pd.DataFrame({"k": ktr, "y": yt})
            agg = df_tmp.groupby("k")["y"].agg(["mean", "size"])
            smooth = (agg["size"] * agg["mean"] + 10 * gm) / (agg["size"] + 10)
            oof[val] = pd.Series(kval).map(smooth).fillna(gm).values
        tr[f"te_{c}"] = oof
        new_cols.append(f"te_{c}")
        ktr_all = tr[c].astype(str).values
        df_tmp = pd.DataFrame({"k": ktr_all, "y": y})
        agg = df_tmp.groupby("k")["y"].agg(["mean", "size"])
        smooth = (agg["size"] * agg["mean"] + 10 * gm) / (agg["size"] + 10)
        te[f"te_{c}"] = te[c].astype(str).map(smooth).fillna(gm).values
    return tr, te, new_cols


def combined(ll, auc):
    return 0.6 * (1 - ll) + 0.4 * auc


def run_lgbm(X, y, Xte, params, return_importance=False):
    """5-fold CV + per-fold isotonic. Returns oof_cal, test_pred, mean_iter
    (+ gain importances summed over folds if return_importance)."""
    oof_cal = np.zeros(len(X))
    test_pred = np.zeros(len(Xte))
    iters = []
    imp = np.zeros(X.shape[1]) if return_importance else None
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for trn, val in skf.split(X, y):
        dtr = lgb.Dataset(X.iloc[trn], y[trn], categorical_feature=CAT_COLS)
        dva = lgb.Dataset(X.iloc[val], y[val], categorical_feature=CAT_COLS,
                          reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        pv = m.predict(X.iloc[val], num_iteration=m.best_iteration)
        pt = m.predict(Xte, num_iteration=m.best_iteration)
        iso = IsotonicRegression(out_of_bounds="clip")
        oof_cal[val] = np.clip(iso.fit_transform(pv, y[val]), 1e-6, 1 - 1e-6)
        test_pred += np.clip(iso.transform(np.clip(pt, 0, 1)), 1e-6, 1 - 1e-6) / N_SPLITS
        iters.append(m.best_iteration)
        if return_importance:
            imp += m.feature_importance(importance_type="gain")
    out = (oof_cal, test_pred, float(np.mean(iters)))
    return out + (imp,) if return_importance else out


def log_row(search_path, step, n_feat, ll, auc, sc, kept):
    pd.DataFrame([{"step": step, "n_features": n_feat, "ll": ll, "auc": auc,
                   "combined": sc, "kept": kept}]).to_csv(
        search_path, mode="a", header=not os.path.exists(search_path),
        index=False)


def save_state(state_path, state):
    with open(state_path, "w") as f:
        json.dump(state, f)


def load_state(state_path):
    if not os.path.exists(state_path):
        return None
    with open(state_path) as f:
        return json.load(f)


def main():
    tr = pd.read_csv(os.path.join(DATA, "Train.csv"))
    te = pd.read_csv(os.path.join(DATA, "Test.csv"))
    tr = add_features_base(tr)
    te = add_features_base(te)
    base_cols = [c for c in tr.columns if c not in (ID_COL, TARGET)]

    bal_p25 = float(np.percentile(tr[BAL_COLS].values.astype(float), 25))
    tr, new_exp = add_features_expanded(tr, bal_p25)
    te, _ = add_features_expanded(te, bal_p25)

    for c in CAT_COLS:
        tr[c] = tr[c].astype("category")
        te[c] = te[c].astype("category")
    y = tr[TARGET].values
    tr, te, new_te = add_target_encoding(tr, te, y)

    feats_all = [c for c in tr.columns if c not in (ID_COL, TARGET)]
    new_feats = [c for c in feats_all if c not in base_cols]
    print(f"base features: {len(base_cols)}, new features: {len(new_feats)}, "
          f"total: {len(feats_all)}", flush=True)
    X, Xte = tr[feats_all], te[feats_all]

    os.makedirs(OUT, exist_ok=True)
    search_path = os.path.join(OUT, "iter6_search.csv")
    state_path = os.path.join(OUT, "iter6_state.json")
    state = load_state(state_path) or {}
    ranked_new = state.get("ranked_new")
    kept_new = state.get("kept_new", [])
    done_batches = set(state.get("done_batches", []))
    best = state.get("best", {"score": -1.0})

    def evaluate(feat_list, imp=False):
        return run_lgbm(tr[feat_list], y, te[feat_list], KITCHEN_SINK,
                        return_importance=imp)

    # Step 0: full model -> gain ranking of new features (once).
    if ranked_new is None:
        print("Step 0: full-model gain ranking...", flush=True)
        oof, _, mean_iter, imp = evaluate(feats_all, imp=True)
        ll = log_loss(y, oof)
        auc = roc_auc_score(y, oof)
        sc = combined(ll, auc)
        gain = pd.Series(imp, index=feats_all).sort_values(ascending=False)
        ranked_new = [f for f in gain.index if f in new_feats]
        print(f"full model ({len(feats_all)} feats): LL={ll:.5f} AUC={auc:.5f} "
              f"comb={sc:.5f}", flush=True)
        log_row(search_path, "full", len(feats_all), ll, auc, sc, False)
        state.update({"ranked_new": ranked_new})
        save_state(state_path, state)
    else:
        print(f"Resuming: loaded gain ranking ({len(ranked_new)} new feats), "
              f"kept so far: {len(kept_new)}, done batches: {sorted(done_batches)}",
              flush=True)

    # Step 1: baseline = base features only.
    if best["score"] < 0:
        oof, _, mean_iter = evaluate(base_cols)
        ll = log_loss(y, oof)
        auc = roc_auc_score(y, oof)
        best = {"score": combined(ll, auc), "ll": ll, "auc": auc}
        print(f"base ({len(base_cols)} feats): LL={ll:.5f} AUC={auc:.5f} "
              f"comb={best['score']:.5f}", flush=True)
        log_row(search_path, "base", len(base_cols), ll, auc, best["score"], True)
        state["best"] = best
        save_state(state_path, state)

    # Step 2: greedy forward in gain-ranked batches.
    batches = [ranked_new[i:i + BATCH] for i in range(0, len(ranked_new), BATCH)]
    fails = 0
    for bi, batch in enumerate(batches):
        if bi in done_batches:
            continue
        if len(base_cols) + len(kept_new) + len(batch) > MAX_FEATURES:
            print(f"batch {bi}: skipped (would exceed {MAX_FEATURES} feats)",
                  flush=True)
            done_batches.add(bi)
            continue
        cand = base_cols + kept_new + batch
        oof, _, mean_iter = evaluate(cand)
        ll = log_loss(y, oof)
        auc = roc_auc_score(y, oof)
        sc = combined(ll, auc)
        kept = sc > best["score"] + EPS
        if kept:
            kept_new.extend(batch)
            best = {"score": sc, "ll": ll, "auc": auc}
            fails = 0
        else:
            fails += 1
        print(f"batch {bi} (+{len(batch)} feats -> {len(cand)} total): "
              f"LL={ll:.5f} AUC={auc:.5f} comb={sc:.5f} "
              f"{'KEPT' if kept else 'dropped'} (best {best['score']:.5f})",
              flush=True)
        log_row(search_path, f"batch{bi}", len(cand), ll, auc, sc, kept)
        done_batches.add(bi)
        state.update({"kept_new": kept_new, "best": best,
                      "done_batches": sorted(done_batches)})
        save_state(state_path, state)
        if fails >= PATIENCE:
            print(f"early stop: {PATIENCE} non-improving batches", flush=True)
            break

    # Step 3: final model on selected set (also yields importances + files).
    final_feats = base_cols + kept_new
    print(f"\nFinal: {len(final_feats)} features "
          f"({len(kept_new)} new kept of {len(new_feats)} tried)", flush=True)
    oof_cal, test_pred, mean_iter, imp = evaluate(final_feats, imp=True)
    ll = log_loss(y, oof_cal)
    auc = roc_auc_score(y, oof_cal)
    sc = combined(ll, auc)
    print(f"FINAL LL={ll:.5f} AUC={auc:.5f} comb={sc:.5f} "
          f"(iter5 bar: {ITER5_BAR}) "
          f"{'BEATS iter5' if sc > ITER5_BAR else 'below iter5 bar'}", flush=True)

    sub = pd.DataFrame({ID_COL: te[ID_COL],
                        "Target": np.clip(test_pred, 1e-6, 1 - 1e-6)})
    sub.to_csv(os.path.join(OUT, "iter6.csv"), index=False)
    pd.DataFrame({"ID": tr[ID_COL], "oof_pred_cal": oof_cal, TARGET: y}).to_csv(
        os.path.join(OUT, "oof_iter6.csv"), index=False)
    print(f"Saved submissions/iter6.csv (test mean {test_pred.mean():.4f}). "
          f"NOT submitted (daily limit reached).", flush=True)

    gain = pd.Series(imp, index=final_feats).sort_values(ascending=False)
    top_new = [(f, g) for f, g in gain.items() if f in new_feats][:10]
    print("Top-10 NEW features by gain:", flush=True)
    for f, g in top_new:
        print(f"  {f:38s} {g:,.0f}", flush=True)
    state.update({"final": {"n_features": len(final_feats),
                            "n_new_kept": len(kept_new),
                            "ll": ll, "auc": auc, "combined": sc,
                            "top10_new": [f for f, _ in top_new]}})
    save_state(state_path, state)


if __name__ == "__main__":
    main()
