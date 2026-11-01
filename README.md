# Financial Stress Prediction Challenge (Zindi, 2026-10-09)

Predict `liquidity_stress_next_30d` (binary) for 30k customers from 6 months of
mobile-money behaviour + customer profile features.

## Data
- `data/Train.csv`: 40,000 rows, 183 features + `ID` + target (15% positive)
- `data/Test.csv`: 30,000 rows, 183 features + `ID`
- Features: profile (`arpu`, `age`, `gender`, `region`, `smartphone`, `segment`,
  `earning_pattern`, 90-day activity rate) + per-month (m1=most recent … m6=oldest)
  behaviour: paybill, merchantpay, transfer_from_bank, mm_send, received,
  deposit, withdraw (volume / total_value / highest_amount / counts) + daily_avg_bal
- No missing values. Categoricals are low-cardinality and consistent train/test.

## Metric
Zindi: 0.6 * LogLoss + 0.4 * ROC-AUC (higher better). We optimize logloss with
early stopping and monitor both; probabilities are clipped to [1e-6, 1-1e-6].

## Approach
`src/train.py`: LightGBM, 5-fold stratified CV (seed 42), native categorical
support, `lr=0.04, num_leaves=63, feature_fraction=bagging_fraction=0.8`,
early stopping (200 rounds) on validation logloss. Test prediction = mean of
the 5 fold models. OOF predictions saved for stacking/calibration.

## Results
| Model | CV LogLoss | CV ROC-AUC | CV combined* |
|---|---|---|---|
| LightGBM baseline (5-fold, seed 42) | 0.27650 | 0.88291 | 0.78727 |

\*combined = 0.6*(1−LogLoss) + 0.4*ROC-AUC (higher better).
Best iterations per fold: 454 / 406 / 347 / 409 / 366.
Top features by gain: `m*_daily_avg_bal` (all 6 months), then
`m*_deposit_total_value` and `m*_received_total_value` — balance trajectory
dominates liquidity-stress signal, as expected. Test prediction mean 0.1319 vs
train positive rate 0.1500 (well calibrated).

## Files
- `src/train.py` — training + CV + submission generation
- `submissions/baseline_lgbm.csv` — `ID,Target` probabilities for test
- `submissions/oof_baseline_lgbm.csv` — out-of-fold predictions on train
