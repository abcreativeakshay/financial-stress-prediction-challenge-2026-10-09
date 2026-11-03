"""Deep residual autopsy on iter14's final 10-seed hybrid blend OOF.
Finds WHERE remaining error lives to guide v6 feature design."""
import os
import numpy as np
import pandas as pd

OUT = os.path.expanduser("~/workspace/financial-stress-prediction-challenge-2026-10-09/submissions")
DATA = os.path.expanduser("~/workspace/financial-stress-prediction-challenge-2026-10-09/data")
EPS = 1e-6
SEEDS10 = [42, 7, 2025, 5, 11, 777, 99, 1234, 2024, 31337]

def hybrid(a, b, w_geo=0.5, w_lgbm=0.5):
    g = np.sqrt(np.clip(a, EPS, 1) * np.clip(b, EPS, 1))
    arith = w_lgbm * a + (1 - w_lgbm) * b
    return w_geo * g + (1 - w_geo) * arith

tr = pd.read_csv(os.path.join(DATA, "Train.csv"))
y = tr["liquidity_stress_next_30d"].values.astype(int)

oofs_l = [np.load(os.path.join(OUT, f"iter14_lgbm_s{s}.npz"))["oof"] for s in SEEDS10]
oofs_c = [np.load(os.path.join(OUT, f"iter14_catboost_s{s}.npz"))["oof"] for s in SEEDS10]
hyb = [hybrid(oofs_l[i], oofs_c[i]) for i in range(10)]
p = np.clip(np.mean(hyb, axis=0), EPS, 1 - EPS)

rowll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
from sklearn.metrics import roc_auc_score
ll = rowll.mean()
auc = roc_auc_score(y, p)
comb = 0.6 * (1 - ll) + 0.4 * auc
print(f"reconstructed blend OOF: LL={ll:.5f} AUC={auc:.5f} combined={comb:.5f} (expect ~0.82543)")

dis = np.std(np.column_stack(oofs_l + oofs_c), axis=1)
dis_lc = np.abs(np.mean(oofs_l, axis=0) - np.mean(oofs_c, axis=0))  # LGBM vs CatBoost disagreement
tr["_p"] = p
tr["_ll"] = rowll
tr["_dis"] = dis
tr["_dislc"] = dis_lc
tr["_resid"] = y - p
tr["_fp_hc"] = (p > 0.7) & (y == 0)
tr["_fn_hc"] = (p < 0.12) & (y == 1)
tr["_hc_wrong"] = tr["_fp_hc"] | tr["_fn_hc"]

def show(df, group_cols, title, min_n=300):
    g = df.groupby(group_cols).agg(
        n=("liquidity_stress_next_30d", "size"),
        pos_rate=("liquidity_stress_next_30d", "mean"),
        mean_ll=("_ll", "mean"),
        hc_rate=("_hc_wrong", "mean"),
        fp_rate=("_fp_hc", "mean"),
        fn_rate=("_fn_hc", "mean"),
        mean_dis=("_dislc", "mean"),
    )
    g = g[g["n"] >= min_n].sort_values("mean_ll", ascending=False)
    print(f"\n== {title} (worst LL first, n>={min_n}) ==")
    print(g.round(4).head(12).to_string())

# 1. categoricals
for c in ["segment", "region", "gender", "smartphone", "earning_pattern"]:
    show(tr, [c], f"by {c}")

# 2. interactions
show(tr, ["segment", "earning_pattern"], "segment x earning_pattern")
show(tr, ["segment", "region"], "segment x region")
show(tr, ["gender", "segment"], "gender x segment")
show(tr, ["smartphone", "segment"], "smartphone x segment")

# 3. binned continuous
for col, nb in [("age", 6), ("arpu", 6), ("x_90_d_activity_rate", 6),
                ("m1_daily_avg_bal", 6), ("m6_daily_avg_bal", 6)]:
    tr["_bin"] = pd.qcut(tr[col], nb, duplicates="drop")
    show(tr, ["_bin"], f"by {col} quantile", min_n=300)

# 4. trajectory-shape slices: balance trend direction
bal = tr[[f"m{i}_daily_avg_bal" for i in range(1, 7)]].values
tr["_bal_trend"] = bal[:, 0] - bal[:, 5]  # m1 - m6
tr["_bal_trend_bin"] = pd.qcut(tr["_bal_trend"], 5, duplicates="drop")
show(tr, ["_bal_trend_bin"], "by balance trend (m1-m6) quintile")
tr["_bal_cv"] = bal.std(axis=1) / (bal.mean(axis=1) + 1)
tr["_balcv_bin"] = pd.qcut(tr["_bal_cv"], 5, duplicates="drop")
show(tr, ["_balcv_bin"], "by balance CV quintile")

# 5. zero-activity structure
vol_cols = [c for c in tr.columns if "_volume" in c]
tr["_zero_months"] = (tr[vol_cols].values == 0).reshape(40000, -1, 7).sum(axis=2).sum(axis=1) if False else None
# per-channel zero months
chans = ["paybill", "merchantpay", "transfer_from_bank", "mm_send", "received", "deposit", "withdraw"]
for ch in chans:
    zc = [f"m{i}_{ch}_volume" for i in range(1, 7)]
    tr[f"_z_{ch}"] = (tr[zc].values == 0).sum(axis=1)
tr["_z_tot"] = tr[[f"_z_{ch}" for ch in chans]].sum(axis=1)
tr["_z_bin"] = pd.cut(tr["_z_tot"], [-1, 5, 15, 25, 42])
show(tr, ["_z_bin"], "by total zero-volume channel-months")

# 6. worst decile profile
worst = tr[rowll > np.quantile(rowll, 0.9)].copy()
print("\n== worst LL decile: positive rate by segment vs overall ==")
print(pd.crosstab(worst["segment"], worst["liquidity_stress_next_30d"], normalize="index").round(3).to_string())
print("\nworst decile: FN share =", (worst["liquidity_stress_next_30d"] == 1).mean().round(4),
      "| overall pos rate =", y.mean().round(4))

# 7. where do LGBM and CatBoost disagree most?
tr["_dis_bin"] = pd.qcut(tr["_dislc"], 5, duplicates="drop")
show(tr, ["_dis_bin"], "by LGBM-vs-CatBoost disagreement quintile")
hi_dis = tr[tr["_dislc"] > np.quantile(tr["_dislc"], 0.9)]
print("\n== top-disagreement decile profile ==")
print("pos_rate:", hi_dis["liquidity_stress_next_30d"].mean().round(4))
print(hi_dis["segment"].value_counts(normalize=True).round(3).to_string())
print(hi_dis["earning_pattern"].value_counts(normalize=True).round(3).to_string())

# 8. false positives: who looks stressed but isn't?
fp = tr[tr["_fp_hc"]].copy()
print(f"\n== high-conf FALSE POSITIVES n={len(fp)} ==")
if len(fp):
    print("segment:", fp["segment"].value_counts(normalize=True).round(3).to_dict())
    print("earning_pattern:", fp["earning_pattern"].value_counts(normalize=True).round(3).to_dict())
    print("median m1_bal:", fp["m1_daily_avg_bal"].median(), "vs overall:", tr["m1_daily_avg_bal"].median())
    print("median activity:", fp["x_90_d_activity_rate"].median(), "vs overall:", tr["x_90_d_activity_rate"].median())

# 9. false negatives: hidden stress the model misses
fn = tr[tr["_fn_hc"]].copy()
print(f"\n== high-conf FALSE NEGATIVES n={len(fn)} ==")
if len(fn):
    print("segment:", fn["segment"].value_counts(normalize=True).round(3).to_dict())
    print("earning_pattern:", fn["earning_pattern"].value_counts(normalize=True).round(3).to_dict())
    print("median m1_bal:", fn["m1_daily_avg_bal"].median(), "vs overall:", tr["m1_daily_avg_bal"].median())
    print("median activity:", fn["x_90_d_activity_rate"].median(), "vs overall:", tr["x_90_d_activity_rate"].median())

# 10. channel divergence: which channel's m1-vs-history disagrees with prediction?
for ch in chans:
    v = tr[[f"m{i}_{ch}_total_value" for i in range(1, 7)]].values
    tr[f"_div_{ch}"] = (v[:, 0] - v[:, 1:].mean(axis=1)) / (v[:, 1:].mean(axis=1) + 1)
tr["_div_bin"] = pd.qcut(tr["_div_withdraw"], 5, duplicates="drop")
show(tr, ["_div_bin"], "by withdraw m1-vs-history divergence")
tr["_div_bin"] = pd.qcut(tr["_div_received"], 5, duplicates="drop")
show(tr, ["_div_bin"], "by received m1-vs-history divergence")
tr["_div_bin"] = pd.qcut(tr["_div_deposit"], 5, duplicates="drop")
show(tr, ["_div_bin"], "by deposit m1-vs-history divergence")

# 11. age x balance interaction (young with high balance? old with low?)
tr["_age_bin"] = pd.qcut(tr["age"], 3, duplicates="drop", labels=["young", "mid", "old"])
tr["_bal_bin"] = pd.qcut(tr["m1_daily_avg_bal"], 3, duplicates="drop", labels=["low", "mid", "high"])
show(tr, ["_age_bin", "_bal_bin"], "age x m1-balance")

# 12. save worst-decile IDs for later inspection
worst_idx = tr.index[rowll > np.quantile(rowll, 0.9)]
tr.loc[worst_idx, ["ID"]].to_csv(os.path.join(OUT, "autopsy_worst_decile_ids.csv"), index=False)
print(f"\nsaved {len(worst_idx)} worst-decile IDs")
