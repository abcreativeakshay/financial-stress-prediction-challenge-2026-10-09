"""Residual/error-slice analysis of iter13's final blend OOF (A_wg0.5_wl0.5)."""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

OUT = os.path.expanduser("~/workspace/financial-stress-prediction-challenge-2026-10-09/submissions")
DATA = os.path.expanduser("~/workspace/financial-stress-prediction-challenge-2026-10-09/data")
EPS = 1e-6
SEEDS6 = [42, 7, 2025, 5, 11, 777]

def hybrid(a, b, w_geo, w_lgbm):
    g = np.sqrt(np.clip(a, EPS, 1) * np.clip(b, EPS, 1))
    arith = w_lgbm * a + (1 - w_lgbm) * b
    return w_geo * g + (1 - w_geo) * arith

tr = pd.read_csv(os.path.join(DATA, "Train.csv"))
y = tr["liquidity_stress_next_30d"].values.astype(int)

oofs_l = [np.load(os.path.join(OUT, f"iter13_lgbm_s{s}.npz"))["oof"] for s in SEEDS6]
oofs_c = [np.load(os.path.join(OUT, f"iter13_catboost_s{s}.npz"))["oof"] for s in SEEDS6]
hyb = [hybrid(oofs_l[i], oofs_c[i], 0.5, 0.5) for i in range(6)]
p = np.clip(np.mean(hyb, axis=0), EPS, 1 - EPS)
rowll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
print(f"blend OOF LL={rowll.mean():.5f}")

# disagreement between members
dis = np.std(np.column_stack(oofs_l + oofs_c), axis=1)
print(f"member-disagreement: mean={dis.mean():.4f} p90={np.quantile(dis, .9):.4f}")

tr["_p"] = p
tr["_ll"] = rowll
tr["_dis"] = dis
tr["_hc_wrong"] = ((p > 0.7) & (y == 0)) | ((p < 0.12) & (y == 1))
tr["_resid"] = y - p

print("\n== high-confidence wrong by segment ==")
print(tr.groupby("segment").agg(n=("segment", "size"), hc_rate=("_hc_wrong", "mean"),
      mean_ll=("_ll", "mean"), pos_rate=("liquidity_stress_next_30d", "mean")).round(4))
print("\n== by region ==")
print(tr.groupby("region").agg(n=("region", "size"), hc_rate=("_hc_wrong", "mean"),
      mean_ll=("_ll", "mean")).round(4))
print("\n== by gender x smartphone ==")
print(tr.groupby(["gender", "smartphone"]).agg(n=("gender", "size"), hc_rate=("_hc_wrong", "mean"),
      mean_ll=("_ll", "mean")).round(4))
print("\n== by earning_pattern ==")
print(tr.groupby("earning_pattern").agg(n=("earning_pattern", "size"), hc_rate=("_hc_wrong", "mean"),
      mean_ll=("_ll", "mean")).round(4))

# continuous slices
for col, nb in [("age", 5), ("arpu", 5), ("x_90_d_activity_rate", 5), ("m1_daily_avg_bal", 5)]:
    tr["_bin"] = pd.qcut(tr[col], nb, duplicates="drop")
    print(f"\n== by {col} quintile ==")
    print(tr.groupby("_bin", observed=True).agg(n=(col, "size"), hc_rate=("_hc_wrong", "mean"),
          mean_ll=("_ll", "mean")).round(4))

# worst decile by row-LL: profile
worst = tr[rowll > np.quantile(rowll, 0.9)]
print("\n== worst LL decile profile vs overall ==")
for c in ["segment", "region", "gender", "smartphone", "earning_pattern"]:
    print(c, "\n", (worst[c].value_counts(normalize=True) - tr[c].value_counts(normalize=True)).round(3).to_dict())
print("\nworst-decile y rate:", worst["liquidity_stress_next_30d"].mean(), " overall:", y.mean())
print("worst-decile mean p:", worst["_p"].mean(), " mean dis:", worst["_dis"].mean())

# false negatives vs false positives structure
fn = tr[(y == 1) & (p < 0.12)]
fp = tr[(y == 0) & (p > 0.7)]
print(f"\nFN (p<0.12,y=1): n={len(fn)} ; FP (p>0.7,y=0): n={len(fp)}")
print("FN segment mix:", fn["segment"].value_counts(normalize=True).round(3).to_dict())
print("FP segment mix:", fp["segment"].value_counts(normalize=True).round(3).to_dict())
print("FN activity median:", fn["x_90_d_activity_rate"].median(), " overall:", tr["x_90_d_activity_rate"].median())
print("FP activity median:", fp["x_90_d_activity_rate"].median())
print("FN m1 bal median:", fn["m1_daily_avg_bal"].median(), " FP m1 bal median:", fp["m1_daily_avg_bal"].median())
print("FN arpu median:", fn["arpu"].median(), " FP arpu median:", fp["arpu"].median())
