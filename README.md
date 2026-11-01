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
| LGBM + feats + OOF target encoding (region/segment/earning_pattern) + isotonic | 0.26287 | 0.89250 | 0.79928 |
| XGBoost + one-hot categoricals + feats + isotonic | 0.26384 | 0.89201 | 0.79850 |
| Stack: 0.55*LGBM + 0.45*XGB, per-model isotonic + blend isotonic | **0.25956** | **0.89549** | **0.80246** |

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
- Public LB (baseline_lgbm.csv): 0.6862. Public LB (iter2.csv): 0.6996.

### Iteration 3 details
- **Stacking** (`src/iter3.py`): 5-fold LGBM (native categoricals) + 5-fold
  XGBoost (one-hot categoricals) on the 216-feature set; per-fold isotonic
  calibration per model, OOF blend weight 0.55 LGBM / 0.45 XGB chosen by
  grid-search on calibrated OOF, plus a final isotonic fit on the blend.
  Honest throughout: target encodings and calibration computed per fold.
- **Target encoding** (`te_region`, `te_segment`, `te_earning_pattern`,
  alpha=20 smoothing, fold-honest): helped LGBM logloss slightly (0.2629 vs
  0.2676 raw) but the blend barely beat the single models — LGBM and XGB
  OOF predictions are highly correlated (models lack diversity).
- **Result**: CV combined 0.80246 — *below* iter2's 0.80415. Recommendation:
  keep iter2.csv as the submission candidate; stacking as implemented did not
  add value. Next ideas: more diverse base learners (CatBoost with native
  categoricals, different seeds/subsamples), feature-set ablation, or
  hyperparameter search rather than more blending.
- Fixed a categorical `.map` bug in `target_encode` (`.astype(object)` first).
- Environment note: VM was replaced mid-task; lightgbm/xgboost/sklearn were
  reinstalled via pip (`--break-system-packages`, TMPDIR workaround for the
  512MB /tmp).

## Files
- `src/train.py` — baseline training + CV + submission generation
- `src/iter2.py` — feature engineering + calibration + model comparison
- `src/iter3.py` — LGBM+XGB stacking with OOF target encoding (CV 0.80246, did not beat iter2)
- `submissions/baseline_lgbm.csv` — `ID,Target` baseline test predictions
- `submissions/oof_baseline_lgbm.csv` — baseline out-of-fold predictions
- `submissions/iter2.csv` — `ID,Target` iteration-2 test predictions (best)
- `submissions/oof_iter2.csv` — iteration-2 calibrated OOF predictions
- `submissions/iter3.csv` — `ID,Target` stacking test predictions (CV 0.80246 < iter2)
- `submissions/oof_iter3.csv` — stacking blend calibrated OOF predictions
