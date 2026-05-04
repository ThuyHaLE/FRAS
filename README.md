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

Custom strategies (e.g., `XGBSEDebiasedBCE`) can be registered by subclassing `ModelStrategy` and implementing `build()`, `fit()`, and `predict_survival()`.

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