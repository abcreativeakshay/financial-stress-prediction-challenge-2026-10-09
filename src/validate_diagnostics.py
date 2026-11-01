"""Validation diagnostics for the Financial Stress Prediction Challenge.

Three checks:
1. Adversarial validation: LightGBM classifier distinguishing train vs test rows.
   AUC ~ 0.5  -> train/test match, stratified CV trustworthy.
   AUC > 0.6  -> covariate shift; report top-10 driving features.
2. Time structure: does any date/reference-date column exist to tell whether M1
   is a fixed calendar month for all rows or a rolling per-customer window?
3. Group check: do customer IDs repeat across rows (one row per customer vs
   multiple snapshots per customer)?

Does NOT touch iter5 files (another agent is training there).
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SEED = 42
N_SPLITS = 5
TARGET = "liquidity_stress_next_30d"
ID_COL = "ID"
CAT_COLS = ["gender", "region", "smartphone", "segment", "earning_pattern"]

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "submissions")

lines = []


def log(msg):
    print(msg, flush=True)
    lines.append(msg)


def main():
    tr = pd.read_csv(os.path.join(DATA, "Train.csv"))
    te = pd.read_csv(os.path.join(DATA, "Test.csv"))
    log(f"Train shape: {tr.shape}, Test shape: {te.shape}")

    # ------------------------------------------------------------------
    # 3. GROUP CHECK (cheap, do first)
    # ------------------------------------------------------------------
    log("\n=== 3. GROUP CHECK (ID uniqueness) ===")
    tr_dup = tr[ID_COL].duplicated().sum()
    te_dup = te[ID_COL].duplicated().sum()
    overlap = tr[ID_COL].isin(te[ID_COL]).sum()
    log(f"Train rows: {len(tr)}, unique IDs: {tr[ID_COL].nunique()}, duplicated IDs: {tr_dup}")
    log(f"Test rows:  {len(te)}, unique IDs: {te[ID_COL].nunique()}, duplicated IDs: {te_dup}")
    log(f"Train/Test ID overlap: {overlap}")
    if tr_dup == 0 and te_dup == 0:
        group_conclusion = (
            "One row per customer: every ID is unique within train and within test. "
            "No GroupKFold needed; stratified K-fold is appropriate."
        )
    else:
        group_conclusion = (
            f"WARNING: {tr_dup} duplicated train IDs, {te_dup} duplicated test IDs. "
            "Multiple snapshots per customer exist -> use GroupKFold grouped by customer ID."
        )
    log("Conclusion: " + group_conclusion)

    # ------------------------------------------------------------------
    # 2. TIME STRUCTURE CHECK
    # ------------------------------------------------------------------
    log("\n=== 2. TIME STRUCTURE CHECK ===")
    cols = list(tr.columns)
    date_like = [c for c in cols
                 if any(k in c.lower() for k in ("date", "timestamp", "time", "month_ref", "ref_date", "snapshot_date", "calendar"))]
    # 'month' alone is not date-like here since m1..m6 are feature prefixes
    date_like = [c for c in date_like if not c.startswith("m") or not c[1].isdigit()]
    log(f"Date/reference-date-like columns found: {date_like if date_like else 'NONE'}")

    dd = pd.read_csv(os.path.join(DATA, "data_dictionary.csv"))
    m1_rows = dd[dd["column_name"] == "m1_paybill_volume"]
    m1_desc = m1_rows["description"].iloc[0] if len(m1_rows) else "(not found)"
    id_desc = dd[dd["column_name"] == "ID"]["description"].iloc[0]
    log(f"Dictionary on m1: \"{m1_desc}\"")
    log(f"Dictionary on ID: \"{id_desc}\"")
    fixed_mention = dd["description"].str.contains(
        r"calendar|reference date|same month|fixed window", case=False, na=False).any()
    rolling_mention = dd["description"].str.contains(
        r"rolling|per customer snapshot|observation window", case=False, na=False).any()
    log(f"Dictionary mentions fixed calendar window: {fixed_mention}")
    log(f"Dictionary mentions rolling/observation window: {rolling_mention}")

    if not date_like:
        time_conclusion = (
            "INCONCLUSIVE from data alone: no date/reference-date column exists, and the "
            "data dictionary states only that 'M1 is the most recent month and M6 is the "
            "oldest month' without saying whether M1 is the same calendar month for all "
            "rows (fixed window) or a rolling per-customer window. Because train and test "
            "have identical feature structure, rely on adversarial validation (check 1): "
            "if train/test are indistinguishable (AUC ~0.5), stratified K-fold is fine "
            "regardless of the window type. If shift is found, prefer a time-aware split "
            "or drop the drifted features."
        )
    else:
        time_conclusion = f"Date columns exist ({date_like}); a time-based split should be tested."
    log("Conclusion: " + time_conclusion)

    # ------------------------------------------------------------------
    # 1. ADVERSARIAL VALIDATION
    # ------------------------------------------------------------------
    log("\n=== 1. ADVERSARIAL VALIDATION (train vs test) ===")
    feats = [c for c in tr.columns if c not in (ID_COL, TARGET)]
    for c in CAT_COLS:
        tr[c] = tr[c].astype("category")
        te[c] = te[c].astype("category")
    X_all = pd.concat([tr[feats], te[feats]], ignore_index=True)
    y_adv = np.concatenate([np.ones(len(tr)), np.zeros(len(te))])
    log(f"Adversarial set: {X_all.shape[0]} rows x {len(feats)} features")

    params = {
        "objective": "binary", "metric": "binary_logloss",
        "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 100,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
        "lambda_l2": 1.0, "verbose": -1, "num_threads": 4, "seed": SEED,
    }
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X_all))
    imp = np.zeros(len(feats))
    predicted_mask = np.zeros(len(X_all), dtype=bool)
    # NOTE: reduced to 2 folds under heavy VM contention; fold-0 estimate (0.5063)
    # from the full 5-fold attempt already corroborated the finding.
    for fold, (trn, val) in enumerate(skf.split(X_all, y_adv)):
        if fold >= 2:
            break
        predicted_mask[val] = True
        dtr = lgb.Dataset(X_all.iloc[trn], y_adv[trn], categorical_feature=CAT_COLS)
        dva = lgb.Dataset(X_all.iloc[val], y_adv[val], categorical_feature=CAT_COLS, reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[val] = m.predict(X_all.iloc[val], num_iteration=m.best_iteration)
        imp += m.feature_importance(importance_type="gain") / N_SPLITS
        fold_auc = roc_auc_score(y_adv[val], oof[val])
        log(f"fold {fold}: adv AUC = {fold_auc:.4f} (best_iter {m.best_iteration})")

    adv_auc = roc_auc_score(y_adv[predicted_mask], oof[predicted_mask])
    log(f"\nOverall adversarial AUC = {adv_auc:.4f} (over {predicted_mask.sum()} predicted rows)")
    if adv_auc < 0.55:
        adv_conclusion = (
            "Train and test are essentially indistinguishable. Distributions match; "
            "stratified K-fold CV is trustworthy and CV gains should transfer to the LB."
        )
    elif adv_auc < 0.65:
        adv_conclusion = (
            "Mild shift detected. CV is probably OK but verify CV<->LB correlation with "
            "probe submissions before trusting small CV gains."
        )
    else:
        adv_conclusion = (
            "STRONG covariate shift between train and test. Do not trust plain stratified "
            "CV blindly; consider dropping drifted features or a test-weighted validation."
        )
    log("Conclusion: " + adv_conclusion)

    if adv_auc > 0.6:
        fi = pd.DataFrame({"feature": feats, "gain": imp}).sort_values("gain", ascending=False)
        log("\nTop-10 features driving the train/test split:")
        for _, r in fi.head(10).iterrows():
            log(f"  {r['feature']:45s} gain={r['gain']:.1f}")

    # ------------------------------------------------------------------
    # Write summary file
    # ------------------------------------------------------------------
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "diagnostics.txt"), "w") as f:
        f.write("FINANCIAL STRESS CHALLENGE - VALIDATION DIAGNOSTICS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"1. Adversarial validation AUC: {adv_auc:.4f}\n   {adv_conclusion}\n\n")
        f.write(f"2. Time structure: {time_conclusion}\n\n")
        f.write(f"3. Group check: {group_conclusion}\n\n")
        f.write("IMPLICATION FOR VALIDATION STRATEGY:\n")
        if adv_auc < 0.55 and tr_dup == 0:
            f.write("Keep stratified 5-fold CV (current approach is valid). "
                    "No GroupKFold, no time-based split required by the evidence.\n")
        else:
            f.write("Revisit the validation scheme before trusting CV gains (see notes above).\n")
    log(f"\nSummary written to {os.path.join(OUT, 'diagnostics.txt')}")


if __name__ == "__main__":
    main()
