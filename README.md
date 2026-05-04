# 🔥 WiDS Global Datathon 2026 — Wildfire Survival Modeling

This repository contains my solution for the [WiDS Global Datathon 2026](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26).

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
 
Features 2–4 are all **temporal coverage** proxies — they reflect how well the fire was observed, not how it behaves. A model trained naïvely on these would largely learn *"fires that are close and well-observed are dangerous"*, which conflates observational bias with actual fire risk.

#### 🛠️ Design Decision

To address this, two deliberate choices were made:
 
1. **`dist_min_ci_0_5h` is removed from `BASE_FEATURES`** before feature selection. Keeping it would cause models to anchor almost entirely on proximity, drowning out the signal from fire dynamics.
```python
remove_feature = ['dist_min_ci_0_5h']
base_features = [f for f in BASE_FEATURES if f not in remove_feature]
```
 
2. **Feature enrichment via `enrich_features()`** synthesizes new features that capture the *physics of fire spread* — ETA estimates, directional threat, growth-distance interactions — using `dist_min_ci_0_5h` as an input but not as a standalone predictor. This way the distance information is still leveraged, but only as part of compound features that encode real threat dynamics.
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
| **Temporal / diurnal** | `is_high_wind_hour`, `hour_sin`, `hour_cos` | Peak fire hours (10:00–18:00) carry elevated spread risk |
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
 
Each model gets its own optimal feature subset, allowing different models to specialize on different signal types.

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
 
This makes it straightforward to experiment with new survival models (e.g. `XGBSEDebiasedBCE`, DeepSurv) or tune `TierThresholds` per model without touching the training or evaluation pipeline.

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
results = ensemble.compare_subsets(train_df, subset_sizes=[2, 3])
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
 
The fitted T is accepted only if the mean calibrated probability stays within 0.5×–2.0× of the base rate. If the ratio falls outside this range (e.g. due to a poor-quality fit or too few positives), T is reset to 1.0 (identity — no calibration). Horizons with fewer than 10 negatives or zero positives are also skipped.
 
At ensemble level, temperature scaling is refit on the blended OOF predictions after weight optimization, so the final calibration reflects the ensemble's combined output rather than individual model outputs.

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
 
### Base Features (34 features)
 
| Model | CV Hybrid Score | OOF Hybrid Score | Public LB |
|---|---|---|---|
| RSF | 0.9315 ± 0.0094 | 0.9290 | 0.89954 |
| GB | 0.9728 ± 0.0041 | 0.9724 | 0.96382 |
| Coxnet | 0.8071 ± 0.1336 | 0.8244 | 0.89657 |
| CGB | 0.8943 ± 0.0148 | 0.8888 | 0.86591 |
| Ensemble 2 models (RSF=0.304, GB=0.696) | — | 0.9701 | 0.96124 |
| Ensemble 4 models (GB=0.804, CGB=0.167, Coxnet=0.029) | — | 0.9709 | 0.96331 |
 
### Enriched Features + Per-Model Feature Selection
 
Selected features per model:
 
**RSF** (14 features): `log1p_dist_min`, `eta_along_track`, `eta_projected`, `eta_closing`, `eta_aligned_along`, `eta_radial`, `eta_combined`, `eta_bearing_adjusted`, `alignment_x_nperim`, `dt_first_last_0_5h`, `alignment_abs`, `dt_x_alignment`, `dist_per_area`, `low_temporal_resolution_0_5h`
 
**GB** (7 features): `log1p_dist_min`, `eta_closing`, `dist_per_area`, `eta_aligned_along`, `dt_first_last_0_5h`, `eta_projected`, `eta_along_track`
 
**Coxnet** (28 features): `log1p_dist_min`, `num_perimeters_0_5h`, `dist_per_area`, `log1p_area_first`, `dt_first_last_0_5h`, `alignment_abs`, `area_speed_ratio`, `eta_closing`, `alignment_x_nperim`, `dist_fit_r2_0_5h`, `dist_cv`, `cross_track_component`, `eta_projected`, `reliable_accel`, `dt_x_alignment`, `dist_change_norm`, `directional_growth`, `log1p_reliable_slope`, `dist_slope_ci_0_5h`, `growth_pressure`, `spread_bearing_sin`, `area_first_ha`, `low_temporal_resolution_0_5h`, `event_start_month`, `reliable_eta_ci`, `spread_bearing_deg`, `linear_eta_ci`, `is_approaching`
 
**CGB** (6 features): `log1p_dist_min`, `eta_closing`, `alignment_abs`, `eta_combined`, `eta_aligned_along`, `cross_track_component`
 
| Model | CV Hybrid Score | OOF Hybrid Score | Public LB |
|---|---|---|---|
| RSF | 0.9737 ± 0.0061 | 0.9711 | 0.96270 |
| GB | 0.9736 ± 0.0063 | 0.9715 | 0.96377 |
| Coxnet | 0.9127 ± 0.0412 | 0.9121 | 0.90317 |
| CGB | 0.9581 ± 0.0085 | 0.9560 | 0.93059 |
| Ensemble 2 models (RSF=0.500, GB=0.500) | — | 0.9689 | **0.96448** |
| Ensemble 4 models (RSF=0.000, GB=0.772, CGB=0.000, Coxnet=0.228) | — | 0.9698 | 0.96253 |
 
### Key Observations
 
**Base features → CV inflated, LB drops (RSF):** RSF on base features achieves CV 0.9315 but drops to 0.8995 on public LB — a gap of ~0.03. This is consistent with the observational bias hypothesis: `dist_min_ci_0_5h` and temporal coverage features produce clean CV signal but do not generalise well.
 
**Enriched features → stable generalisation:** After replacing raw distance with compound physics-based features, RSF CV improves to 0.9737 and LB closes to 0.9627 — CV/LB gap shrinks from ~0.03 to ~0.01. This validates the design decision to remove `dist_min_ci_0_5h` from base features and re-encode distance as ETA and threat features.
 
**Coxnet and CGB underperform consistently:** Despite feature selection, both models lag significantly behind RSF and GB — Coxnet by ~0.06 LB points and CGB by ~0.03. Coxnet also shows high CV variance (±0.04), indicating instability. Adding them to the ensemble does not improve LB: on base features the optimizer nearly zeros out both (Coxnet=0.029, CGB=0.167, RSF=0.000); on enriched features Coxnet receives a more substantial weight (0.228), yet the 4-model ensemble still scores 0.9625 vs 0.9645 for RSF+GB only — indicating that Coxnet does not add complementary signal but instead dilutes GB's predictions.
 
**Ensemble → lower CV, higher LB:** The final RSF+GB ensemble OOF score (0.9689) is slightly below individual models, but public LB (0.9645) exceeds both. Blending reduces individual model overfit and calibration corrects the probability scale — consistent with the classic CV/LB trade-off of ensemble methods.
 
**Limitations and future directions:** Feature enrichment improves all models on CV, but the gains do not translate equally across models or to ensemble diversity. A few specific observations:
 
**GB enrichment is marginal:** GB base (LB 0.96382) and GB enriched (LB 0.96377) are nearly identical, suggesting GB already saturates on the core ETA + distance signal and enrichment adds little beyond what the base features capture.
  
**Linear models struggle with non-linear features:** The engineered features are predominantly non-linear transformations (ETA ratios, log-compressed speeds, trigonometric bearing terms) — tree-based models (RSF, GB) exploit these directly, but Coxnet requires 28 features after selection yet still underperforms, likely because these transformations do not align with its linear proportional-hazard assumption.

**Feature selection thresholds are fixed and may not be optimal:** `TierThresholds` uses fixed importance cutoffs (`star=0.005`, `tier2=0.0005`, `noise_ratio=5.0`) tuned for small datasets. Since Coxnet's permutation importance scale is ~10–30× larger than tree-based models, it uses separate `COXNET_THRESHOLDS` — but both are manually set and could benefit from cross-validated threshold search.

**`enrich_features` is a fixed pipeline:** All feature groups are computed unconditionally and several internal constants (`SLOPE_THRESHOLD = -0.1`, `MAX_ETA_HOURS = 500`, `EPS = 0.01`) are hardcoded magic numbers. These could be tuned — for instance, tightening `SLOPE_THRESHOLD` to filter out noisy ETA estimates, or adding new feature groups (e.g. weather × terrain interactions) — without changing anything else in the pipeline.

A natural next step would be to include non-linear models with different inductive biases — such as `XGBSEDebiasedBCE` (already implemented in this repo) or neural survival models (e.g. DeepSurv) — which may capture interaction patterns that RSF and GB miss, and could add genuine diversity to the ensemble beyond what Coxnet/CGB offer.

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
├── FRAS_demo.ipynb              # web-based demo version
├── requirements.txt
└── README.md
```

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

---

## 📅 Timeline

* Start: Jan 28, 2026
* Entry Deadline: Apr 24, 2026
* Final Submission: May 1, 2026