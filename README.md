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
| LightGBM baseline (5-fold, seed 42, raw 183 feats) | 0.27650 | 0.88291 | 0.78727 |
| LightGBM base params + 40 engineered feats + per-fold isotonic calibration | **0.25788** | **0.89720** | **0.80415** |
| LightGBM tuned params (lr 0.03, leaves 127) + feats + isotonic | — | — | < 0.80415 |
| XGBoost (hist, 5-fold) + feats + isotonic | 0.26079 | 0.89475 | 0.80143 |

\*combined = 0.6*(1−LogLoss) + 0.4*ROC-AUC (higher better).

### Iteration 2 details
- **Feature engineering** (`src/iter2.py::add_features`, +40 features): balance
  trajectory over m1–m6 `daily_avg_bal` (slope, std, min, max, mean, m1/m6
  ratio, m1−m6, max month-over-month drop, # of drops); per activity group
  (deposit, withdraw, received, mm_send, paybill, merchantpay,
  transfer_from_bank) total_value sums + recent(m1+m2) vs old(m5+m6) ratios;
  net monthly flow (deposit+received − withdraw−mm_send) sum/slope/m1;
  withdraw/deposit ratio.
- **Calibration**: isotonic regression fit per-fold on validation predictions
  (honest — never on training data), applied to that fold's test predictions
  then averaged. Fixed the baseline's under-confidence at the high end
  (top decile: pred 0.61 vs actual 0.70). Platt scaling was tried and rejected
  (hurt logloss: 0.2855).
- **Model comparison**: LightGBM base params won; tuned LightGBM and XGBoost
  (note: xgboost warned `enable_categorical` unused — categoricals likely fell
  back to integer codes) both scored lower.
- Test prediction mean 0.1520 vs train positive rate 0.1500 (well calibrated).
- Public LB (baseline_lgbm.csv): 0.6862.

## Files
- `src/train.py` — baseline training + CV + submission generation
- `src/iter2.py` — feature engineering + calibration + model comparison
- `submissions/baseline_lgbm.csv` — `ID,Target` baseline test predictions
- `submissions/oof_baseline_lgbm.csv` — baseline out-of-fold predictions
- `submissions/iter2.csv` — `ID,Target` iteration-2 test predictions (best)
- `submissions/oof_iter2.csv` — iteration-2 calibrated OOF predictions
