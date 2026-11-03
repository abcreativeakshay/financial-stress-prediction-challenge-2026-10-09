# V6 Plan — Financial Stress Prediction Challenge

Date: 2026-11-03. Context: iter15's v5 battery failed (0/79 kept by greedy OOF
selection); the 20-seed blend finishing ~10-11 PM IST tonight is seed-gains only
(CV ~0.826-0.828, public ~0.725-0.728 projected). Leader 0.74293, gap ~0.018.
V6 is the next real lever. Battery code ready at `src/v6_battery.py` (107
candidates, all finite/non-constant, train/test distributions sane).

## 1. Zindi forum research

Status: research agent dispatched; findings to be appended here.
(Preliminary: no public notebooks found yet for this October-2026 edition;
ideas below are derived from our own autopsy. Update when forum report lands.)

## 2. Residual autopsy — iter14's 10-seed hybrid blend OOF (CV 0.82543)

Reconstructed exactly (LL 0.23551, AUC 0.91684, combined 0.82543). Method:
per-seed hybrid `0.5*sqrt(lgbm*cb) + 0.5*(0.5*lgbm+0.5*cb)`, averaged over
10 seeds. Row log-loss sliced across segments, regions, quantiles, and
trajectory shapes.

### Where the error lives
1. **HVC segment is the hardest pocket**: mean row-LL 0.3214 vs 0.2048 (MVC)
   and 0.2552 (LVC). Worst cells: HVC×Western Zone 0.3486 (n=408),
   HVC×Northern Highlands 0.3406, HVC×Monthly Earner 0.3228. HVC is only
   9.3% of rows but contributes disproportionately. LGBM-vs-CatBoost
   disagreement is also highest on HVC (0.0235).
2. **ARPU U-shape (NEW)**: top arpu decile LL 0.2678 AND bottom decile 0.2526;
   middle deciles ~0.20-0.21. Both affluent and poor users are mispredicted —
   the model treats arpu monotonically and misses both tails.
3. **Balance volatility**: top balance-CV quintile LL 0.2902 vs 0.1818 lowest.
4. **Inflow cliffs**: sharp m1 collapse in received (LL 0.2685) and deposit
   (0.2736) are among the worst behavioral pockets.
5. **Zero-activity U-shape**: 25-42 zero channel-months → 0.2627; 5-15 → 0.2527;
   middle (15-25, the bulk) → 0.2173.
6. **Worst decile is 75.4% false negatives** — the model systematically
   UNDER-predicts stress. False negatives have HIGHER median m1 balance
   (5840 vs 5082 overall): hidden stress behind decent balances.
7. **LGBM/CatBoost disagreement is itself predictive**: top disagreement
   quintile has 38.7% positive rate (vs 15% base) and LL 0.4956. The members
   make different errors; a smarter combination than plain averaging may exist.
8. **Diversity trends correlate with target** (raw, pre-model):
   deposit-agents slope +0.105, received-senders slope +0.073,
   withdraw-agents slope −0.066, banks slope +0.044.
   Interpretation: stressed users show EXPANDING deposit-agent/sender networks
   (scrambling for inflows from more sources) but CONTRACTING withdrawal-agent
   networks (less cash-out). The 42 network-diversity columns had ZERO
   engineered usage in v1-v5.

### Why v5 failed
v5 (79 candidates: inact×X, seg×X, accelerations, robust z-scores) was the
same mask×ratio/slope family as v4. Greedy OOF selection kept 0/79 — the model
already spans that subspace. v6 uses fundamentally different transforms.

## 3. V6 battery design (107 candidates, `src/v6_battery.py`)

All features unsupervised (no target use). Quantile cutoffs and peer stats fit
on TRAIN, applied to test. Tested: all finite, non-constant, sane tr/te stats.

### Family A — network-diversity dynamics (41; fresh: 0 prior usage)
- slopes + m1-vs-history ratios for 7 diversity metrics (paybill companies,
  merchants, banks, mm-send recipients, received senders, deposit agents,
  withdraw agents)
- value-per-counterparty (avg ticket per relationship): m1 + slope × 7
- `v6_div_scramble` = depag_slope − wdag_slope (the scramble signature)
- `v6_div_senders_up_recv_down` (more senders, falling received value)
- total network breadth: m1, slope, CV

### Family B — peer-relative / arpu-conditioned (7; attacks the arpu U-shape)
- arpu decile (train quantiles → test)
- within-decile z-scores of balance/outflow/inflow/activity (train medians/IQRs)
- `v6_rank_mismatch_bal_arpu` (balance percentile − arpu percentile)
- outflow vs arpu-decile peer median

### Family C — time-series shape (18; fresh beyond slopes/CVs)
- normalized Shannon entropy of 6-month distributions (bal/inflow/outflow/net)
- # direction changes, longest monotonic run
- **time-since-peak** (0 = m1 is peak) — strongest univariate in battery
- late-vs-early half ratios

### Family D — ticket-size / transaction structure (22; fresh)
- avg ticket (value/volume) m1 + slope × 7 channels
- concentration (highest_amount/total_value) m1 × 7
- volume-value divergence

### Family E — cliff detectors (14; fresh binary formulation)
- m1 < 0.3×median(m2..m6) for received/deposit/withdraw/paybill/inflow/balance,
  each also × HVC
- sudden-stop flags (m1=0 after active history) for received/deposit

### Family F — channel substitution & bank channel (7; fresh)
- paybill-share-of-bills slope (channel migration), cash-vs-digital share m1+slope
- bank top-up slope/share, net bank flow (transfer_from_bank − withdraw)

### Family G — hard-regime flags (5; fresh)
- HVC × volatile / sparse / low-activity (HVC-relative thresholds)
- hidden stress: ok-balance + inflow cliff; high-arpu + inactive

### Univariate signal check (|corr| with target, train)
Top: `v6_shape_bal_t_since_peak` +0.177, `v6_shape_inflow_t_since_peak` +0.167,
`v6_shape_net_t_since_peak` +0.156, inflow entropy −0.124, `v6_div_scramble`
+0.121, bal entropy −0.114, depag slope +0.105, peer inflow z −0.098.
Family ranking by top-5 mean |corr|: shape 0.148 ≫ div 0.086 > suddenstop 0.072
> peer 0.059 > cliff 0.047 > ticket 0.040. (Univariate ≠ CV gain, but the shape
family's "time since peak" is unlike anything in v1-v5.)

## 4. The 3 most promising v6 directions

1. **Time-since-peak + shape features (Family C).** Rationale: corr +0.177 is
   the strongest univariate signal we've seen from any engineered feature; it
   captures "declining from a previous high" — a natural liquidity-stress
   precursor that slopes/CVs/maxdd don't isolate (a user can have a gentle
   negative slope but peaked 5 months ago, vs peaked last month — very
   different stress implications). Entropy adds distributional shape the trees
   can't easily reconstruct. If any single family breaks the v5 curse, it's this.
2. **Network-diversity dynamics (Family A).** Rationale: 42 raw columns with
   zero engineered usage; diversity slopes correlate +0.105/+0.073/−0.066 with
   the target; the "scramble" signature (expanding inflow network + contracting
   outflow network) is a behavioral story, not a statistical artifact. Trees see
   the raw levels but not the trends — that's the gap.
3. **Peer-relative / arpu-conditioned features (Family B).** Rationale: the
   autopsy found a clean arpu U-shape the model misses; conditioning behavior
   on arpu-decile peers ("is this user unusual FOR THEIR wealth level?")
   directly attacks it. Rank-mismatch (balance rank vs arpu rank) is a compact
   way to flag users living above/below their means.

## 5. Execution plan (when iter15 blend finishes ~10-11 PM IST)

1. Run greedy forward OOF selection of the 107 v6 candidates on the iter15
   399-feature base (same protocol as v1-v5: seed-42 LGBM+CatBoost hybrid,
   keep-if-beats-bar). Expect the shape family to survive first.
2. If kept features beat 0.82543: rebuild full 20-seed hybrid on 399+v6-kept,
   submit as iter16 (9 daily slots remain for Nov 3; quota resets 05:30 IST).
3. If v6 also fails: next levers are (a) member-disagreement exploitation —
   the top-disagreement quintile is 38.7% positive, suggesting a learned
   nonlinear combiner; (b) target-encoded aggregates with proper OOF handling;
   (c) forum ideas (pending).

## 6. Files
- `src/v6_battery.py` — battery code (tested, ready)
- `src/autopsy14.py` — residual autopsy script
- `submissions/autopsy_worst_decile_ids.csv` — worst-decile row IDs
- `submissions/v6_plan.md` — this doc
