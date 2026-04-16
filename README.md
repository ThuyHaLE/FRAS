# 🔥 WiDS Global Datathon 2026 — Wildfire Survival Modeling

This repository contains my solution for the WiDS Global Datathon 2026.

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
- event==0 means confirmed NOT hit within 72h window (closed window, not traditional censoring)
- Therefore ALL records have known outcomes at every horizon → no exclusion mask needed
- max_time in train ≈ 67h < 72h, but event==0 records are still valid negatives at 72h

---

## 🧠 Approach (Draft)

### 1. Labeling Strategy

---

### 2. Feature Engineering

---

### 3. Models

---

### 4. Calibration

---

## 📏 Evaluation

Models are evaluated using Hybrid Score = 0.3 x C-index + 0.7 x (1 - Weighted Brier Score)

* **C-index (30%)** (ranking quality)
* **Weighted Brier Score (70%)** (calibration): `0.3 x Brier@24h + 0.4 x Brier@48h + 0.3 x Brier@72h`

---

## 🗂️ Repository Structure

---

## 🚀 How to Run

---

## 📅 Timeline

* Start: Jan 28, 2026
* Entry Deadline: Apr 24, 2026
* Final Submission: May 1, 2026

---
