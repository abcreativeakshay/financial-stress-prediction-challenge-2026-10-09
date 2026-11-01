"""Iteration 5: combine iter4's winners + try CatBoost.

Part A (cheap, likely stacks): iter4's winner leaves_31 (CV 0.80501) combined
with the other regularizers that helped singly: lambda_l2=3.0 (0.80484),
lambda_l1=0.5 (0.80469), plus lr=0.03 (0.80439).
  - leaves31_l2   : num_leaves=31 + lambda_l2=3.0          -> 0.80512 (done)
  - leaves31_l1   : num_leaves=31 + lambda_l1=0.5          -> 0.80529 (done)
  - leaves31_l1l2 : num_leaves=31 + lambda_l2=3.0 + lambda_l1=0.5
  - kitchen_sink  : num_leaves=31 + lambda_l2=3.0 + lambda_l1=0.5 + lr=0.03

Part B: CatBoost with [gender, region, smartphone, segment, earning_pattern]
as categoricals, same CV scheme and per-fold isotonic calibration.

Everything kept identical to iter4 for comparability:
- Same 40 engineered features (add_features copied verbatim)
- 5-fold stratified CV, seed 42 (deterministic -> reruns reproduce metrics)
- Per-fold isotonic calibration (fit on validation preds only)
- Metric: combined = 0.6*(1-LogLoss) + 0.4*ROC-AUC on calibrated OOF

RESUME-SAFE: per-config scores appended to submissions/iter5_search.csv as
the search runs. On restart, completed configs are skipped; if the overall
winner is a skipped config, it is rerun once at the end to materialize its
OOF/test predictions for the submission files.

Winner (must beat iter4's 0.80501 to be submittable) -> submissions/iter5.csv,
oof_iter5.csv.
"""
import os
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
ITER4_BAR = 0.80501  # only submit if we beat this

BAL_COLS = [f"m{i}_daily_avg_bal" for i in range(6, 0, -1)]  # m6 oldest .. m1 newest
GROUPS = ["deposit", "withdraw", "received", "mm_send", "paybill",
          "merchantpay", "transfer_from_bank"]

LGBM_BASE = {
    "objective": "binary", "metric": "binary_logloss",
    "learning_rate": 0.04, "num_leaves": 63, "min_child_samples": 40,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "verbose": -1, "num_threads": 4, "seed": SEED,
}

# (name, param overrides off BASE)
LGBM_SEARCH = [
    ("leaves31_l2", {"num_leaves": 31, "lambda_l2": 3.0}),
    ("leaves31_l1", {"num_leaves": 31, "lambda_l1": 0.5}),
    ("leaves31_l1l2", {"num_leaves": 31, "lambda_l2": 3.0, "lambda_l1": 0.5}),
    ("kitchen_sink", {"num_leaves": 31, "lambda_l2": 3.0, "lambda_l1": 0.5,
                      "learning_rate": 0.03}),
]

CATBOOST_DESC = "depth=6,lr=0.03,l2=5.0,iters=4000,od=300"


def add_features(df):
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


def combined(ll, auc):
    return 0.6 * (1 - ll) + 0.4 * auc


def run_lgbm(X, y, Xte, params):
    """5-fold CV with per-fold isotonic calibration. Returns oof_cal, test_pred, mean_iter."""
    oof_cal = np.zeros(len(X))
    test_pred = np.zeros(len(Xte))
    iters = []
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
    return oof_cal, test_pred, float(np.mean(iters))


def run_catboost(X, y, Xte, cat_idx):
    """5-fold CV CatBoost + per-fold isotonic. Returns oof_cal, test_pred, mean_iter."""
    from catboost import CatBoostClassifier, Pool
    oof_cal = np.zeros(len(X))
    test_pred = np.zeros(len(Xte))
    iters = []
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for trn, val in skf.split(X, y):
        ptr = Pool(X.iloc[trn], y[trn], cat_features=cat_idx)
        pva = Pool(X.iloc[val], y[val], cat_features=cat_idx)
        pte = Pool(Xte, cat_features=cat_idx)
        m = CatBoostClassifier(
            iterations=4000, learning_rate=0.03, depth=6,
            l2_leaf_reg=5.0, loss_function="Logloss", eval_metric="AUC",
            random_seed=SEED, od_type="Iter", od_wait=300,
            thread_count=4, verbose=False, allow_writing_files=False,
        )
        m.fit(ptr, eval_set=pva, use_best_model=True)
        pv = m.predict_proba(X.iloc[val])[:, 1]
        pt = m.predict_proba(pte)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        oof_cal[val] = np.clip(iso.fit_transform(pv, y[val]), 1e-6, 1 - 1e-6)
        test_pred += np.clip(iso.transform(np.clip(pt, 0, 1)), 1e-6, 1 - 1e-6) / N_SPLITS
        iters.append(m.tree_count_)
    return oof_cal, test_pred, float(np.mean(iters))


def log_row(search_path, name, ll, auc, sc, mean_iter, extra):
    pd.DataFrame([{"config": name, "ll": ll, "auc": auc,
                   "combined": sc, "mean_iter": mean_iter,
                   "overrides": str(extra)}]).to_csv(
        search_path, mode="a", header=not os.path.exists(search_path),
        index=False)


def load_done(search_path):
    """name -> (ll, auc, combined) for already-completed configs."""
    if not os.path.exists(search_path):
        return {}
    df = pd.read_csv(search_path)
    df = df[~df["config"].str.endswith("_FAILED")]
    return {r["config"]: (r["ll"], r["auc"], r["combined"])
            for _, r in df.iterrows()}


def main():
    tr = pd.read_csv(os.path.join(DATA, "Train.csv"))
    te = pd.read_csv(os.path.join(DATA, "Test.csv"))
    tr = add_features(tr)
    te = add_features(te)
    for c in CAT_COLS:
        tr[c] = tr[c].astype("category")
        te[c] = te[c].astype("category")
    feats = [c for c in tr.columns if c not in (ID_COL, TARGET)]
    print(f"n_features: {len(feats)}", flush=True)
    X, y, Xte = tr[feats], tr[TARGET].values, te[feats]
    cat_idx = [feats.index(c) for c in CAT_COLS]

    os.makedirs(OUT, exist_ok=True)
    search_path = os.path.join(OUT, "iter5_search.csv")
    done = load_done(search_path)
    if done:
        print(f"Resuming: skipping completed configs {sorted(done)}", flush=True)

    best = {"name": None, "score": -1.0}
    for name, (ll, auc, sc) in done.items():
        if sc > best["score"]:
            best = {"name": name, "score": sc, "ll": ll, "auc": auc}
    store = {}
    skipped_winner = best["name"]  # may need rerun at end for predictions

    print("=== Part A: LGBM winner combos ===", flush=True)
    for name, over in LGBM_SEARCH:
        if name in done:
            print(f"config {name:14s} SKIPPED (done, comb={done[name][2]:.5f})",
                  flush=True)
            continue
        params = dict(LGBM_BASE)
        params.update(over)
        oof_cal, test_pred, mean_iter = run_lgbm(X, y, Xte, params)
        ll = log_loss(y, oof_cal)
        auc = roc_auc_score(y, oof_cal)
        sc = combined(ll, auc)
        store[name] = (oof_cal, test_pred)
        print(f"config {name:14s} LL={ll:.5f} AUC={auc:.5f} comb={sc:.5f} "
              f"mean_iter={mean_iter:.0f}", flush=True)
        log_row(search_path, name, ll, auc, sc, mean_iter, over)
        if sc > best["score"]:
            best = {"name": name, "score": sc, "ll": ll, "auc": auc}

    print("=== Part B: CatBoost ===", flush=True)
    if "catboost" in done:
        print(f"config {'catboost':14s} SKIPPED (done, comb={done['catboost'][2]:.5f})",
              flush=True)
    else:
        try:
            oof_cal, test_pred, mean_iter = run_catboost(X, y, Xte, cat_idx)
            ll = log_loss(y, oof_cal)
            auc = roc_auc_score(y, oof_cal)
            sc = combined(ll, auc)
            store["catboost"] = (oof_cal, test_pred)
            print(f"config {'catboost':14s} LL={ll:.5f} AUC={auc:.5f} comb={sc:.5f} "
                  f"mean_iter={mean_iter:.0f}", flush=True)
            log_row(search_path, "catboost", ll, auc, sc, mean_iter,
                    CATBOOST_DESC)
            if sc > best["score"]:
                best = {"name": "catboost", "score": sc, "ll": ll, "auc": auc}
        except Exception as e:
            print(f"CatBoost FAILED: {e}", flush=True)
            log_row(search_path, "catboost_FAILED", np.nan, np.nan, np.nan,
                    np.nan, str(e)[:200])

    # Materialize predictions for the winner (rerun if it was skipped).
    if best["name"] not in store:
        print(f"Winner '{best['name']}' was skipped; rerunning once for "
              f"predictions...", flush=True)
        if best["name"] == "catboost":
            oof_cal, test_pred, _ = run_catboost(X, y, Xte, cat_idx)
        else:
            over = dict(LGBM_SEARCH)[best["name"]]
            params = dict(LGBM_BASE)
            params.update(over)
            oof_cal, test_pred, _ = run_lgbm(X, y, Xte, params)
        store[best["name"]] = (oof_cal, test_pred)

    print(f"\nBEST: {best['name']} LL={best['ll']:.5f} AUC={best['auc']:.5f} "
          f"comb={best['score']:.5f}  (iter4 bar: {ITER4_BAR})", flush=True)

    oof_cal, test_pred = store[best["name"]]
    sub = pd.DataFrame({ID_COL: te[ID_COL],
                        "Target": np.clip(test_pred, 1e-6, 1 - 1e-6)})
    sub.to_csv(os.path.join(OUT, "iter5.csv"), index=False)
    pd.DataFrame({"ID": tr[ID_COL], "oof_pred_cal": oof_cal, TARGET: y}).to_csv(
        os.path.join(OUT, "oof_iter5.csv"), index=False)
    print(f"Saved submissions/iter5.csv via config '{best['name']}' "
          f"(test mean {test_pred.mean():.4f}); "
          f"{'BEATS iter4 -> submittable' if best['score'] > ITER4_BAR else 'below iter4 bar -> do NOT submit'}",
          flush=True)


if __name__ == "__main__":
    main()
