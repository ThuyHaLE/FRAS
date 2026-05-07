# 🔥 WiDS Global Datathon 2026 — Wildfire Survival Modeling

This repository contains my solution for the [WiDS Global Datathon 2026](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26). It is **FRAS — Fire Risk Assessment System**.

**💡 Why a dashboard, not just model predictions?**

The model outputs calibrated probabilities (`prob_12h / 24h / 48h / 72h`) — but a single number is rarely enough for operational decision-making.

A few structural limitations worth noting:

- **The model does not use all available features.** After permutation-based selection, each model works with a small subset (6–28 features). Signals like surge acceleration, flanking drift, or off-hours response constraints may carry real operational weight but contribute little to the learned survival function.
- **Training data quality shapes what the model learns.** Temporal coverage bias means the model partially learns *"well-observed fires near zones are dangerous"* rather than pure fire spread physics. A fire with sparse perimeter data may receive a low probability, not because it is safe, but because the model had little signal to work with.
- **Two fires with identical `prob_24h` can require completely different responses.** One may be a slow, large fire that has already consumed most of its distance buffer; the other a small, fast-moving fire entering a blowup phase 15 km away. The probability alone does not distinguish them.

The dashboard addresses this by computing **interpretable alarm signals directly from raw feature values** — independent of what the model learned. This creates a transparency layer where:

- Alarm signals surface the *mechanism* behind a prediction (proximity, surge, trajectory confidence, data sparsity, resource availability)
- Divergence between alarms and model probability is itself informative — it reveals which features the model is not relying on for a given fire
- The parameter editor lets users probe feature sensitivity interactively, effectively discovering model behavior without reading SHAP values or importance reports

The goal is not to replace the model output but to give emergency managers enough context to act on it — and to know when to trust it.

---

## 📌 Overview

When a wildfire ignites, emergency managers must make critical decisions with limited early information:
 
* Which fires will become dangerous?
* How soon will they threaten populated areas?
* Which incidents should be prioritized first?

This competition frames the problem as a **survival analysis task**, where the goal is to predict:
 
> The probability that a wildfire will reach within 5 km of an evacuation zone within specific time horizons.

---

## 🎯 Objective

Build models that output **calibrated probabilities** for:
 
* `12 hours`
* `24 hours`
* `48 hours`
* `72 hours`
  
using only data from the **first 5 hours after ignition**.
 
Key requirements:
 
* ✅ Good **ranking performance** (which fire is more urgent)
* ✅ Well **calibrated probabilities** (reliable risk estimates)

---

## 📊 Problem Formulation

This is a **right-censored survival analysis problem**:
 
* `event = 1`: fire reaches evacuation zone within 72h
* `event = 0`: no hit observed within 72h (censored)
* `time_to_hit_hours`: time from `t0 + 5h` until event (or censoring)

Instead of predicting exact time, the model estimates: `P(T≤H), H ∈ {12,24,48,72}`
 
Dataset note:
- `event==0` means confirmed NOT hit within 72h window (closed window, not traditional censoring)
- `max_time` in train ≈ 67h < 72h, but `event==0` records are still valid negatives at 72h
- However, `event==0` records with `time_to_hit < H` are excluded at that horizon (censored before horizon), so a NaN mask is applied per-horizon in the Brier score computation

---

## 🧠 Approach
 
### 1. Labeling Strategy
 
Binary labels are derived from the survival formulation at each time horizon. For each record and each horizon H ∈ {12, 24, 48, 72}:
 
```
y_H = 1    if event == 1  and  time_to_hit ≤ H   (hit before horizon)
y_H = 0    if event == 0  and  time_to_hit ≥ H   (confirmed safe at horizon)
y_H = 0    if event == 1  and  time_to_hit >  H   (hit after horizon)
y_H = NaN  if event == 0  and  time_to_hit <  H   (censored before horizon — excluded)
```
 
The `NaN` case handles **right-censored records**: if a fire was only observed up to time T < H with no hit, we cannot confirm it was safe at H, so it is excluded from the Brier score at that horizon. In practice, since `event==0` in this dataset means a **confirmed non-hit within the closed 72h window**, censoring only occurs at sub-72h horizons (e.g., a fire observed for 30h with no hit is censored at H=48h and H=72h from a strict survival standpoint, but is a valid negative at H=12h and H=24h).
 
This logic is implemented in `make_binary_outcome_matrix()` and `brier_matrix()`, which apply the mask `hit_before | safe_at_h | hit_after` and set excluded cells to `NaN` before computing `nanmean`.
 
---

### 2. Feature Engineering

#### 🔍 EDA Insight

[`EDA`](data/WiDS_2026_EDA.ipynb) revealed that the top-5 most predictive raw features are dominated by **data quality indicators** and **raw distance to the evacuation zone**, not by intrinsic fire behavior:
 
| Rank | Feature | Correlation w/ event | Description |
|------|---------|----------------------|-------------|
| 1 | `dist_min_ci_0_5h` | 0.481 | Min distance to nearest evac zone centroid (m) |
| 2 | `low_temporal_resolution_0_5h` | 0.379 | Flag: sparse perimeter data (dt < 0.5h or only 1 perimeter) |
| 3 | `num_perimeters_0_5h` | 0.371 | Number of perimeters captured in first 5h |
| 4 | `dt_first_last_0_5h` | 0.353 | Time span between first and last perimeter (h) |
| 5 | `alignment_abs` | 0.349 | Absolute alignment of fire spread toward evac zone |
 
Features 2–4 are all **temporal coverage** proxies — they reflect how well the fire was observed, not how it behaves. A model trained naïvely on these would largely learn *"fires that are close and well-observed are dangerous,"* which conflates observational bias with actual fire risk.

#### 🛠️ Design Decision

To address this, two deliberate choices were made:
 
1. **`dist_min_ci_0_5h` is removed from `BASE_FEATURES`** before feature selection. Keeping it would cause models to anchor almost entirely on proximity, drowning out the signal from fire dynamics.
```python
remove_feature = ['dist_min_ci_0_5h']
base_features = [f for f in BASE_FEATURES if f not in remove_feature]
```
 
2. **Feature enrichment via `enrich_features()`** synthesizes new features that capture the *physics of fire spread* — ETA estimates, directional threat, growth-distance interactions — using `dist_min_ci_0_5h` as an input but not as a standalone predictor. This way, the distance information is still leveraged, but only as part of compound features that encode real threat dynamics.
```python
enriched_train_df = enrich_features(pd.read_csv("data/train.csv"))
enriched_features = list(set(enriched_train_df.columns) - set(train_df.columns))
```

#### 📐 Engineered Feature Groups

| Group | Examples | Intuition |
|-------|----------|-----------|
| **ETA estimates** | `eta_closing`, `eta_radial`, `eta_combined`, `eta_aligned_along` | How many hours until fire reaches evac zone at current speed |
| **Directional threat** | `alignment_abs`, `bearing_threat_factor`, `threat_alignment`, `cross_track_ratio` | Is the fire actually heading toward the evac zone? |
| **Growth-distance compound** | `growth_pressure`, `area_radial_threat`, `directional_growth` | Fast-growing + close fires = high risk |
| **Log-transformed speeds** | `log1p_closing_speed`, `log1p_radial_growth_rate`, `log1p_centroid_speed` | Stabilise skewed speed distributions |
| **Slope quality** | `reliable_slope`, `linear_eta_ci`, `reliable_eta_ci` | Weight ETA estimates by R² fit quality |
| **Temporal/diurnal** | `is_high_wind_hour`, `hour_sin`, `hour_cos` | Peak fire hours (10:00–18:00) carry elevated spread risk |
| **Momentum proxies** | `momentum_threat`, `dt_x_alignment`, `alignment_x_nperim` | Sustained directional movement toward CI |

---

### 3. Feature Selection via Permutation Importance

After enrichment, `PermutationImportance` is used to identify the most predictive features for each model type. Features are ranked and bucketed into tiers (TIER1 / TIER2) based on importance thresholds.
 
**Single-model selection:**
```python
result = filter_features(
    enriched_train_df,
    base_features,
    enriched_features,
    model_type="rsf",
    n_repeats=10,
)
survival_features = result["survival_features"]  # TIER1 + TIER2
```
 
**Multi-model independent selection:**
```python
multi = filter_features_independent(
    enriched_train_df,
    base_features,
    enriched_features,
    n_repeats=10,
)
# → {"rsf": [...], "gradboost": [...], "coxnet": [...], "cgb": [...]}
```
 
Each model gets its own optimal feature subset, allowing different models to specialize in different signal types.

---

### 4. Models

Four survival analysis strategies are used, each injected via a `ModelStrategy` interface:

| Model | Class | Notes |
|---|---|---|
| RandomSurvivalForest | `RSFStrategy` | Tree ensemble, robust to non-linearity |
| GradientBoostingSurvivalAnalysis | `GradBoostStrategy` | Boosted survival trees |
| CoxnetSurvivalAnalysis | `CoxnetStrategy` | Regularized Cox PH, sparse features |
| ComponentwiseGradientBoostingSurvivalAnalysis | `CGBStrategy` | Categorical-aware boosting |

#### 🔌 Extensibility
 
The entire pipeline — training, feature selection, and ensemble — is designed to accept any model without modifying core logic. To plug in a new model:
 
**1. Implement `ModelStrategy`:**
```python
class XGBSEStrategy(ModelStrategy):
    @property
    def name(self) -> str: return "XGBSE"
 
    def build(self, random_state: int): ...
    def fit(self, model, X, y): ...
    def predict_survival(self, model, X) -> list: ...
```
 
**2. Run feature selection with custom thresholds:**
```python
result = filter_features(
    enriched_train_df,
    base_features,
    enriched_features,
    strategy=XGBSEStrategy(),
    thresholds=TierThresholds(star=0.005, tier2=0.0005, noise_ratio=4.0),
    n_repeats=10,
)
xgb_features = result["survival_features"]
```
 
**3. Drop into the ensemble — no other changes needed:**
```python
configs = [
    TrainerConfig("RSF",  RSFStrategy(),   multi["rsf"]),
    TrainerConfig("GB",   GradBoostStrategy(), multi["gradboost"]),
    TrainerConfig("XGBSE", XGBSEStrategy(), xgb_features),  # ← new
]
ensemble = EnsembleTrainer(configs)
```
 
This makes it straightforward to experiment with new survival models (e.g., `XGBSEDebiasedBCE`, DeepSurv) or tune `TierThresholds` per model without touching the training or evaluation pipeline.

---

### 5. Ensemble

Models are combined via a weighted ensemble optimized on the training set. Feature subsets come from the multi-model selection in Step 3:
 
```python
# multi = filter_features_independent(enriched_train_df, base_features, enriched_features, n_repeats=10)
# → {"rsf": [...], "gradboost": [...], "coxnet": [...], "cgb": [...]}
 
configs = [
    TrainerConfig("RSF",    RSFStrategy(n_trees=300),  multi["rsf"]),
    TrainerConfig("GB",     GradBoostStrategy(),       multi["gradboost"]),
    TrainerConfig("Coxnet", CoxnetStrategy(),          multi["coxnet"]),
    TrainerConfig("CGB",    CGBStrategy(),             multi["cgb"]),
]
 
ensemble = EnsembleTrainer(configs)
ensemble.cross_validate(train_df)
ensemble.optimize_weights(train_df, strategy="scipy")
 
# Optional: compare sub-ensembles to find the best combination
results = ensemble.compare_subsets(train_df, subset_sizes=[2, 3, 4])
ensemble.apply_best_subset(results["best"], train_df)
 
ensemble.fit(train_df)
df_sub = ensemble.make_submission(test_df)
```
 
Weights are optimized to maximize the Hybrid Score on held-out folds.

---

### 6. Calibration

Survival probabilities at each horizon are calibrated post-hoc using **temperature scaling** — a per-horizon scalar T fitted by minimizing negative log-likelihood on OOF predictions:
 
```
p_cal = sigmoid(logit(p_raw) / T),   T ∈ [0.5, 10.0]
```
 
The fitted T is accepted only if the mean calibrated probability stays within 0.5×–2.0× of the base rate. If the ratio falls outside this range (e.g., due to a poor-quality fit or too few positives), T is reset to 1.0 (identity — no calibration). Horizons with fewer than 10 negatives or zero positives are also skipped.
 
At the ensemble level, temperature scaling is refit on the blended OOF predictions after weight optimization, so the final calibration reflects the ensemble's combined output rather than individual model outputs.

---

## 📏 Evaluation

Models are evaluated using:
 
```
Hybrid Score = 0.3 × C-index + 0.7 × (1 − Weighted Brier Score)
```
 
* **C-index (30%)** — ranking quality: does the model rank more urgent fires higher? Since `max_time` in train ≈ 67h < 72h, C-index is evaluated at **@48h** instead of @72h to avoid extrapolation beyond the observed time range.
* **Weighted Brier Score (70%)** — calibration quality:
```
WBS = 0.3 × Brier@24h + 0.4 × Brier@48h + 0.3 × Brier@72h
```

> **Note:** `@12h` is predicted by the model but excluded from WBS. The metric focuses on the 24–72h window where operational value is highest: `@48h` carries the most weight as it balances actionable lead time with decision urgency; `@72h` is included for extended planning but is less operationally immediate; `@12h` offers too short a lead time to meaningfully inform evacuation decisions.

---
 
## 📈 Results

Five feature strategies were evaluated to understand the contribution of distance bias, feature selection, and enrichment:

### 1. Only `dist_min_ci_0_5h` (1 feature)

| Model | CV Hybrid Score | OOF Hybrid Score | Public LB | Private LB |
|---|---|---|---|---|
| RSF | 0.9634 ± 0.0125 | 0.9645 | 0.94223 | 0.92047 |
| GB | 0.9620 ± 0.0120 | 0.9631 | 0.93615 | 0.93530 |
| Coxnet | 0.9540 ± 0.0130 | 0.9547 | 0.94172 | 0.94422 |
| CGB | 0.8949 ± 0.0171 | 0.8763 | 0.89244 | 0.91058 |

### 2. Base Features (34 features, no selection)

| Model | CV Hybrid Score | OOF Hybrid Score | Public LB | Private LB |
|---|---|---|---|---|
| RSF | 0.9315 ± 0.0094 | 0.9290 | 0.89954 | 0.89710 |
| GB | 0.9728 ± 0.0041 | 0.9724 | 0.96382 | 0.94672 |
| Coxnet | 0.8071 ± 0.1336 | 0.8244 | 0.89657 | 0.92199 |
| CGB | 0.8943 ± 0.0148 | 0.8888 | 0.86591 | 0.89453 |
| RSF + GB manual test (RSF=0.304, GB=0.696) | — | 0.9701 | 0.96124 | 0.94471 |
| Ensemble 4 models — top 1/5: 3/4 (GB=0.804, CGB=0.167, Coxnet=0.029) | — | 0.9709 | 0.96331 | 0.95710 |

### 3. Base Features + Per-Model Feature Selection

Selected features per model:
- **RSF** (9): `dist_min_ci_0_5h`, `event_start_dayofweek`, `alignment_abs`, `dt_first_last_0_5h`, `num_perimeters_0_5h`, `low_temporal_resolution_0_5h`, `spread_bearing_cos`, `alignment_cos`, `spread_bearing_deg`
- **GB** (8): `dist_min_ci_0_5h`, `num_perimeters_0_5h`, `dt_first_last_0_5h`, `log1p_area_first`, `alignment_abs`, `spread_bearing_cos`, `alignment_cos`, `low_temporal_resolution_0_5h`
- **Coxnet** (20): `dist_min_ci_0_5h`, `num_perimeters_0_5h`, `spread_bearing_sin`, `spread_bearing_deg`, `dist_accel_m_per_h2`, `area_growth_rate_ha_per_h`, `closing_speed_abs_m_per_h`, `dist_fit_r2_0_5h`, `dist_slope_ci_0_5h`, `centroid_speed_m_per_h`, `spread_bearing_cos`, `area_first_ha`, `along_track_speed`, `cross_track_component`, `area_growth_abs_0_5h`, `low_temporal_resolution_0_5h`, `projected_advance_m`, `dist_change_ci_0_5h`, `log1p_growth`, `dt_first_last_0_5h`
- **CGB** (6): `dist_min_ci_0_5h`, `num_perimeters_0_5h`, `low_temporal_resolution_0_5h`, `alignment_abs`, `cross_track_component`, `spread_bearing_deg`

| Model | CV Hybrid Score | OOF Hybrid Score | Public LB | Private LB |
|---|---|---|---|---|
| RSF | 0.9739 ± 0.0054 | 0.9724 | **0.96530** | 0.92680 |
| GB | 0.9737 ± 0.0067 | 0.9726 | 0.96322 | 0.94517 |
| Coxnet | 0.9068 ± 0.0212 | 0.9056 | 0.91824 | 0.93849 |
| CGB | 0.8956 ± 0.0132 | 0.8907 | 0.86645 | 0.89543 |
| RSF + GB manual test (RSF=0.501, GB=0.499) | — | 0.9711 | 0.96452 | 0.93618 |
| Ensemble 4 models — top 1/5: 2/4 (GB=0.827, CGB=0.173) | — | 0.9713 | 0.96267 | 0.95402 |

### 4. Base Features + Enriched Features — all 80 features, no selection (removes `dist_min_ci_0_5h`)

| Model | CV Hybrid Score | OOF Hybrid Score | Public LB | Private LB |
|---|---|---|---|---|
| RSF | 0.9699 ± 0.0156 | 0.9677 | 0.94411 | 0.92439 |
| GB | 0.9715 ± 0.0045 | 0.9703 | 0.96090 | 0.95728 |
| Coxnet | 0.8999 ± 0.0314 | 0.9022 | 0.89194 | 0.91666 |
| CGB | 0.9554 ± 0.0091 | 0.9534 | 0.93125 | 0.95508 |
| RSF + GB manual test (RSF=0.492, GB=0.508) | — | 0.9693 | 0.96051 | 0.94890 |
| Ensemble 4 models — top 1/5: 2/4 (RSF=0.492, GB=0.508) | — | 0.9693 | 0.96051 | 0.94890 |

> Note: In Case 4, the manual RSF+GB test and the pipeline's top 
> ensemble produced identical weights and scores — the optimizer 
> independently converged to the same configuration.

### 5. Enriched Features + Per-Model Feature Selection (removes `dist_min_ci_0_5h`) ✅ Final

Selected features per model:
- **RSF** (14): `log1p_dist_min`, `eta_along_track`, `eta_projected`, `eta_closing`, `eta_aligned_along`, `eta_radial`, `eta_combined`, `eta_bearing_adjusted`, `alignment_x_nperim`, `dt_first_last_0_5h`, `alignment_abs`, `dt_x_alignment`, `dist_per_area`, `low_temporal_resolution_0_5h`
- **GB** (7): `log1p_dist_min`, `eta_closing`, `dist_per_area`, `eta_aligned_along`, `dt_first_last_0_5h`, `eta_projected`, `eta_along_track`
- **Coxnet** (28): `log1p_dist_min`, `num_perimeters_0_5h`, `dist_per_area`, `log1p_area_first`, `dt_first_last_0_5h`, `alignment_abs`, `area_speed_ratio`, `eta_closing`,   `alignment_x_nperim`, `dist_fit_r2_0_5h`, `dist_cv`, `cross_track_component`, `eta_projected`, `reliable_accel`, `dt_x_alignment`, `dist_change_norm`, `directional_growth`,   `log1p_reliable_slope`, `dist_slope_ci_0_5h`, `growth_pressure`, `spread_bearing_sin`, `area_first_ha`, `low_temporal_resolution_0_5h`, `event_start_month`, `reliable_eta_ci`, `spread_bearing_deg`, `linear_eta_ci`, `is_approaching`
- **CGB** (6): `log1p_dist_min`, `eta_closing`, `alignment_abs`, `eta_combined`, `eta_aligned_along`, `cross_track_component`

| Model | CV Hybrid Score | OOF Hybrid Score | Public LB | Private LB |
|---|---|---|---|---|
| RSF | 0.9737 ± 0.0061 | 0.9711 | 0.96270 | 0.93674 |
| GB | 0.9736 ± 0.0063 | 0.9715 | 0.96377 | 0.95588 |
| Coxnet | 0.9127 ± 0.0412 | 0.9121 | 0.90317 | 0.94538 |
| CGB | 0.9581 ± 0.0085 | 0.9560 | 0.93059 | 0.95585 |
| RSF + GB manual test (RSF=0.500, GB=0.500) | — | 0.9689 | 0.96448 | 0.95008 |
| Ensemble 4 models — top 1/5: 4/4 (RSF=0.330, GB=0.330, Coxnet=0.098, CGB=0.242) | — | 0.9705 | 0.95966 | 0.95571 |
| Ensemble 4 models — top 2/5: 2/4 (GB=0.772, Coxnet=0.228) ✅ | — | 0.9698 | 0.96253 | **0.95847** |

---

### 📊 Public vs Private Gap — Summary

The table below tracks how well each configuration generalises from public to private LB. A smaller gap indicates more robust generalisation.

| Case | Configuration | Public LB | Private LB | Gap |
|------|--------------|-----------|------------|-----|
| 3 | RSF single (base + filter) | 0.96530 | 0.92680 | -0.039 |
| 3 | RSF + GB manual test | 0.96452 | 0.93618 | -0.030 |
| 2 | RSF + GB manual test | 0.96124 | 0.94471 | -0.017 |
| 5 | RSF + GB manual test | 0.96448 | 0.95008 | -0.014 |
| 3 | Ensemble 4 models top 1/5 | 0.96267 | 0.95402 | -0.009 |
| 2 | Ensemble 4 models top 1/5 | 0.96331 | 0.95710 | -0.006 |
| 5 | Ensemble 4 models top 1/5: 4/4 | 0.95966 | 0.95571 | -0.004 |
| 5 | Ensemble 4 models top 2/5: GB+Coxnet | 0.96253 | **0.95847** | **-0.004** |

The RSF+GB manual test consistently achieves high public LB but larger public-private gaps across all cases — the two strongest individual models do not automatically form the most robust ensemble. The pipeline's `compare_subsets` reliably finds configurations that generalise better, even when they exclude RSF entirely.

---

### Key Observations

**`dist_min_ci_0_5h` alone is a surprisingly strong predictor:** A single-feature RSF (Case 1) reaches public LB 0.942 — comparable to many multi-feature baselines. This confirms that proximity dominates the prediction signal and can mask the contribution of all other features.

**Base features without selection → CV inflated, LB unstable (Case 2):** RSF drops from CV 0.9315 to public LB 0.8995 (~0.03 gap). With all 34 features, including `dist_min_ci_0_5h`, the model anchors on proximity and temporal coverage bias without learning fire dynamics.

**Base features + filter → highest single-model public LB, but confirmed fragile on private (Case 3):** RSF reaches public LB **0.96530** — the highest single-model score across all cases. But private LB drops to **0.92680**, a gap of 0.039. This is the clearest evidence in the results that learning a compact proximity-and-observation-quality signal does not generalise: the public/private split exposed exactly the distribution shift that was anticipated.

**Enriched features without selection → too much noise (Case 4):** Feeding all 80 features without filtering hurts RSF (public 0.944) and does not meaningfully benefit GB (public 0.961). Without feature selection, the enriched space introduces more noise than signal for tree models. Notably, the pipeline's top ensemble converged to the same RSF+GB weights as the manual test — an independent confirmation that these two models dominate the signal in the unfiltered enriched space.

**Enriched features + filter → most robust generalisation, with a public/private tradeoff in the ensemble (Case 5 ✅):** The final approach removes the proximity shortcut and builds on ETA estimates, directional threat, and growth-distance interactions — features grounded in fire spread physics.

Three ensemble configurations were compared:

- **RSF + GB manual test:** highest public LB (0.96448) but lower private LB (0.95008) — consistent with the pattern seen across all cases where the two strongest individual models do not form the most robust ensemble
- **Pipeline top 1/5 — 4/4 models:** highest CV score (0.9705) but lower public (0.95966) and private LB (0.95571) — adding all models introduces noise rather than a complementary signal
- **Pipeline top 2/5 — GB + Coxnet ✅:** best balance — lower public LB (0.96253) but highest private LB across all cases (**0.95847**)

The public/private divergence reflects a distribution shift between the two leaderboard splits. CV score aligned better with private LB than public LB did — but this cannot be known in advance without seeing the private results.

**Coxnet: public LB understates actual performance.** Across all cases, Coxnet consistently scores lower on public LB than on private — for example, in Case 5: public 0.903 vs private **0.945**. This suggests Coxnet's regularized linear structure generalises well, but happens to underfit whatever subset of fires the public split contains. It is not 
a weak model — it is a model whose public score is a poor proxy for its true generalisation.

**CGB shows the same pattern in enriched features (Case 4):** CGB public 0.931 vs private **0.955** — a gap of 0.024 in the opposite direction of what was seen with tree models in Case 3. Combined with Coxnet's behaviour, this suggests the public/private split has a systematic bias that favours RSF and GB over linear and component-wise models.

**RSF+GB manual test vs pipeline ensemble — a consistent pattern:** Across all cases, the manual RSF+GB test scores higher on public LB but generalises worse to private LB than the pipeline's best subset. This is a useful reminder that optimizing for the two individually strongest models is not the same as optimizing for ensemble diversity — and that `compare_subsets` adds real value beyond simply combining the top performers.

**Limitations and future directions:** Feature enrichment improves all models on CV, but the gains do not translate equally across models or to ensemble diversity. A few specific observations:

- **GB enrichment is marginal:** GB base (LB 0.96382) and GB enriched (LB 0.96377) are nearly identical, suggesting GB already saturates on the core ETA + distance signal and enrichment adds little beyond what the base features capture.

- **Coxnet's public LB understates its actual generalisation:** The engineered features are predominantly non-linear transformations (ETA ratios, log-compressed speeds, trigonometric bearing terms) — Coxnet's linear proportional-hazard assumption means it requires 28 features after selection and scores poorly on public LB. However, 
  private LB tells a different story (Case 5: public 0.903 vs private 0.945) — Coxnet generalises well and contributes a complementary signal in the best ensemble (GB+Coxnet, private 0.95847). The limitation is not model quality but public LB reliability as a proxy for Coxnet's true performance.

- **Adding all 4 models hurts rather than helps:** The full 4-model ensemble in Case 5 achieved a higher CV (0.9705) but lower public and private LB than the best 2-model subset. This suggests that despite having distinct feature subsets, RSF and CGB do not add signal that complements GB+Coxnet on the private distribution — the diversity exists in feature space but not in prediction space.

- **Feature selection thresholds are fixed and may not be optimal:** `TierThresholds` uses fixed importance cutoffs (`star=0.005`, `tier2=0.0005`, `noise_ratio=5.0`) tuned for small datasets. Since Coxnet's permutation importance scale is ~10–30× larger than tree-based models, it uses separate `COXNET_THRESHOLDS` — but both are manually set and could benefit from cross-validated threshold search, particularly to improve RSF and CGB subset quality and enable genuine 4-model diversity.

- **`enrich_features` is a fixed pipeline:** All feature groups are computed unconditionally, and several internal constants (`SLOPE_THRESHOLD = -0.1`, `MAX_ETA_HOURS = 500`, `EPS = 0.01`) are hardcoded. These could be tuned — for instance, tightening `SLOPE_THRESHOLD` to filter out noisy ETA estimates, or adding new feature groups (e.g., weather × terrain interactions) — without changing anything else in the pipeline.

A natural next step would be to include models with different inductive biases — such as `XGBSEDebiasedBCE` (already implemented in this repo) or neural survival models (e.g., DeepSurv) — which may capture interaction patterns that RSF and GB miss, and could enable a genuinely diverse 4-model ensemble rather than the optimizer consistently favouring 2-model subsets.

---
 
## 🖥️ Interactive Risk Dashboard (FRAS Dashboard)
 
The repository includes a React-based dashboard (`dashboard/`) that goes well beyond displaying raw model probabilities. While the model outputs `prob_12h / 24h / 48h / 72h`, the dashboard layers on a set of **interpretable alarm signals** computed directly from input features — giving emergency managers actionable insight into *why* a fire is risky, not just *how likely* it is to reach a zone.
 
### 1. Why alarms complement model predictions
 
Beyond the limitations outlined above, the alarm system addresses a practical gap: probabilities rank fires by urgency but do not explain the mechanism driving that urgency. 

The signals below are designed to fill that gap — each capturing a distinct behavioral axis that the model uses internally but does not surface.
 
### 2. Module-level alarms
 
Each tab in the dashboard computes an **overall alarm score ∈ [0, 1]** and a level (`OK` / `WATCH` / `ALARM`) from the features in that module:
 
| Module | Alarm | What it captures |
|--------|-------|-----------------|
| **Centroid Kinematics** | `useCkAlarm` | Fire centroid speed and total displacement — whether the fire is actively moving across terrain. Includes a fast-path that triggers on speed alone, before displacement accumulates. |
| **Fire Growth** | `useFgAlarm` | Absolute size, growth rate (sqrt-scaled), and relative burst (log area ratio) — distinguishes slow large fires from fast small ones. |
| **Directionality** | `useDirAlarm` | Bearing alignment toward evac zones, cross-track drift, and along-track approach speed. Gated: if `alignment_cos < 0.15` (fire moving roughly perpendicular), the alarm is suppressed entirely. |
| **Proximity** | `useRsAlarm` | Distance-to-zone composite (proximity + projected advance), approach rate weighted by R² confidence, and acceleration signal. Negative `dist_accel` (fire speeding up toward zones) is surfaced as a distinct sub-signal. |
 
All normalization ceilings come from `FEATURE_RANGES` — no hardcoded magic numbers — so alarm scores stay calibrated when ranges are updated from new data.
 
### 3. Cross-module scenario alarms
 
The **Alarms tab** (Overview panel) synthesises signals across modules into 13 higher-level scenarios. Each scenario is designed to capture a distinct behavioral or operational axis that no single module alarm covers:
 
| # | Scenario | Question answered | Key signals |
|---|----------|-------------------|-------------|
| 1 | **Data sparsity risk** | Can we trust the other alarms right now? | Temporal coverage flag, R² noise, perimeter count, observation span |
| 2 | **Containment difficulty** | How hard will this fire be to suppress? | Growth rate + 24h reach probability + size + proximity |
| 3 | **Projected advance risk** | How much of the safety buffer has been consumed? | Observed advance ÷ remaining distance (buffer ratio) + closing speed |
| 4 | **Time-to-reach estimate** | How soon could fire reach the nearest evac zone? | Effective closing speed (centroid speed × alignment) → ETA in hours |
| 5 | **Ignition timing risk** | Did the fire start when environmental conditions favor spread? | Diurnal window (RH trough/peak wind hours) × growth rate × detection quality |
| 6 | **Off-hours response risk** | Are suppression resources constrained by time/day of week? | Night hours (aerial ops unavailable) × weekend staffing × growth urgency |
| 7 | **Relative growth intensity** | Is this fire exploding relative to its starting size? | Log area ratio + radial expansion + relative fraction — catches small fires that have 5–10× multiplied |
| 8 | **Surge / blowup risk** | Is the fire entering a runaway phase right now? | Area growth acceleration + closing acceleration + centroid speed, discounted by data quality |
| 9 | **Directional spread pressure** | How much force is the fire exerting toward zones? | Radial expansion × bearing alignment × fire size |
| 10 | **Front fragmentation** | Is this a coherent single front or a multi-flank event? | Perimeter distance spread (σ) × radial width × relative growth |
| 11 | **Approach consistency** | Is the closing signal steady or erratic? | Closing slope gates the score — R² and front coherence only amplify if fire is actually approaching |
| 12 | **Trajectory confidence** | How reliably is the fire closing in on zones? | R² fit quality × negative slope × closing acceleration × data density |
| 13 | **Flanking threat** | Is the fire threatening zones off its main axis? | Lateral drift × radial expansion, size-amplified — small fires have limited flank reach regardless |
 
Scenario ordering is intentional: **data quality first** (scenario 1 primes the reader to weight downstream signals appropriately), then operational impact, ignition context, fire behavior, and directional threat.
 
### 4. Parameter editor & live prediction
 
The dashboard includes a collapsible **Param Editor** panel (available on the Overview tab) that lets power users:
 
- Adjust any feature value via slider or numeric input in real time
- Derived features (log transforms, bearing sin/cos, `low_temporal_resolution` flag) are auto-computed as dependencies change
- Click **▶ Predict** to send the current feature state to the `/api/predict` endpoint and update all alarm signals and probability displays simultaneously
  
This makes it straightforward to test hypothetical scenarios — e.g., "what happens to the alarms if closing speed doubles?" or "how does the trajectory confidence change when R² drops from 0.8 to 0.3?" — without touching any model code.
 
A separate **Upload & Predict** modal accepts a raw JSON event object, runs prediction, and hydrates all dashboard panels at once.

---

## 🚀 How to Run

**1. Clone and install:**
```bash
git clone https://github.com/ThuyHaLE/FRAS.git
cd FRAS
pip install -r requirements.txt
```
 
**2. Quick single-model run:**
```python
import pandas as pd
from features import BASE_FEATURES
from models import SurvivalTrainer
 
train_df = pd.read_csv("data/train.csv")
test_df  = pd.read_csv("data/test.csv")
 
trainer = SurvivalTrainer(feature_cols=BASE_FEATURES, model_type="rsf")
trainer.cross_validate(train_df)
trainer.fit(train_df)
 
df_sub = trainer.make_submission(test_df)
```
 
**3. Full pipeline (enrich → select → ensemble):**

See [`quickstart`](models/quickstart.ipynb) for the complete end-to-end walkthrough.

**4. Interactive dashboard demo:**
```bash
jupyter notebook FRAS_demo.ipynb
```
Runs a self-contained notebook that launches the full FRAS dashboard — all alarm signals, cross-module scenarios, and the parameter editor — without needing a separate frontend build step.

---

## 🗂️ Repository Structure

```
FRAS/
├── data/
│   ├── train.csv
│   └── test.csv
├── features/
│   ├── __init__.py
│   ├── base.py                  # BASE_FEATURES, HORIZONS, BRIER_W, CINDEX_HORIZON, RANDOM_STATE
│   └── target.py                # TARGET_TIME, TARGET_EVENT
├── models/
│   ├── __init__.py              # SurvivalTrainer, EnsembleTrainer, TrainerConfig
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py              # ModelStrategy
│   │   └── builtin.py           # RSFStrategy, GradBoostStrategy, ...
│   ├── ensemble_trainer.py
│   ├── survival_model.py
│   ├── quickstart.ipynb         # End-to-end usage examples
│   └── utils/
│       ├── enrich_features.py   # Feature engineering pipeline
│       └── filter_features.py   # PermutationImportance, filter_features, TierThresholds
├── FRAS_demo.ipynb              # interactive dashboard demo — run to launch full FRAS UI
├── requirements.txt
└── README.md
```

---

## 📅 Timeline

* Competition start: Jan 28, 2026
* Entry Deadline: Apr 24, 2026
* Final Submission: May 1, 2026
  
* Personal: built end-to-end from Apr 16 to May 6, 2026
* Final result: Public LB 0.96253 · Private LB 0.95847
