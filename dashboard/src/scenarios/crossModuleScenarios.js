// dashboard/src/scenarios/crossModuleScenarios.js
// Composite scenarios that integrate signals from multiple modules to provide
// higher-level insights about fire behavior and risk — complementing model-predicted
// prob_12/24/48/72h with interpretable, feature-driven alarm signals.
//
// Each scenario returns:
//   { title, subtitle, sources, accentColor, overallScore, signals, summary }
//
// Design principles:
//   - overallScore ∈ [0, 1], produced by clamp01()
//   - alarmLevel(score) → "ok" | "watch" | "alarm"
//   - Signals are orthogonal: each scenario captures a distinct behavioral axis
//   - Amplifier pattern: secondary signals scale a primary signal, never dominate alone
//   - Negative feature values (retreat, shrink) clamp to 0 without breaking hint strings

import { clamp01, alarmLevel } from "../utils/alarmUtils";
import { R } from "../constants/featureRanges";

// ─── HELPERS ────────────────────────────────────────────────────────────────

/** Safe non-negative value — prevents negative inputs from producing misleading hints */
const nn = (x) => Math.max(x ?? 0, 0);

/** Format meters/hour speed into a readable string */
const fmtSpeed = (mph) => `${mph.toFixed(1)} m/h`;

/** Format km with one decimal */
const fmtKm = (m) => `${(m / 1000).toFixed(1)} km`;


// ─── 1. TIME-TO-REACH ESTIMATE ───────────────────────────────────────────────
// Question: How soon could this fire reach the nearest evac zone?
// Primary axis: ETA in hours based on effective closing speed and alignment.

export function scenarioETA(ckVals, rsVals) {
  const v = { ...ckVals, ...rsVals };

  const effectiveSpeed = Math.max(
    v.centroid_speed_m_per_h * Math.max(v.alignment_cos ?? 0, 0),
    v.closing_speed_m_per_h
  );
  const etaH       = effectiveSpeed > 0.5 ? v.dist_min_ci_0_5h / effectiveSpeed : null;
  const etaScore   = effectiveSpeed > 0.5 ? clamp01(1 - etaH / 72) : 0;
  const alignScore = clamp01(((v.alignment_cos ?? 0) - 0.2) / 0.8);
  const speedScore = clamp01(v.centroid_speed_m_per_h / R("centroid_speed_m_per_h").max);
  const combined   = clamp01(etaScore * 0.6 + alignScore * 0.25 + speedScore * 0.15);
  const etaLabel   = etaH === null
    ? "N/A (stationary)"
    : etaH < 1
      ? `~${Math.round(etaH * 60)} min`
      : `~${etaH.toFixed(1)} h`;
  const level = alarmLevel(combined);

  return {
    title: "Time-to-reach estimate",
    subtitle: "centroid speed × bearing alignment → effective approach toward nearest evac zone",
    sources: ["centroid_speed_m_per_h", "alignment_cos", "dist_min_ci_0_5h", "closing_speed_m_per_h"],
    accentColor: "#378ADD",
    overallScore: combined,
    signals: [
      { name: "effective closing speed", value: speedScore,  hint: `${fmtSpeed(effectiveSpeed)} (speed × alignment)` },
      { name: "alignment toward zone",   value: alignScore,  hint: `alignment_cos = ${(v.alignment_cos ?? 0).toFixed(2)} → ${(v.alignment_cos ?? 0) > 0.7 ? "direct" : (v.alignment_cos ?? 0) > 0.3 ? "oblique" : "tangential"}` },
      { name: "ETA estimate",            value: etaScore,    hint: `${etaLabel} at current effective speed` },
    ],
    summary: {
      ok:    `Fire is not approaching evac zones at this time. ETA: ${etaLabel}.`,
      watch: `Fire may reach zones in ${etaLabel} if current bearing holds.`,
      alarm: `Rapid approach — ETA ${etaLabel}. Immediate action window.`,
    }[level],
  };
}


// ─── 2. DIRECTIONAL SPREAD PRESSURE ──────────────────────────────────────────
// Question: How much force is the fire exerting toward evac zones right now?
// Primary axis: Radial expansion rate weighted by bearing alignment and fire size.

export function scenarioSpreadPressure(ckVals, fgVals, dirVals) {
  const v         = { ...ckVals, ...fgVals, ...dirVals };
  const radialNorm = clamp01(v.radial_growth_rate_m_per_h / R("radial_growth_rate_m_per_h").max);
  const alignPos   = clamp01(v.alignment_cos);
  const areaScale  = clamp01(Math.log1p(v.area_first_ha) / R("log1p_area_first").max);
  const pressure   = clamp01(radialNorm * alignPos * (0.5 + 0.5 * areaScale));
  const level      = alarmLevel(pressure);

  return {
    title: "Directional spread pressure",
    subtitle: "radial expansion rate × bearing alignment × fire size → net push toward evac zones",
    sources: ["radial_growth_rate_m_per_h", "alignment_cos", "area_first_ha", "spread_bearing_deg"],
    accentColor: "#D85A30",
    overallScore: pressure,
    signals: [
      { name: "radial expansion rate", value: radialNorm, hint: `${fmtSpeed(v.radial_growth_rate_m_per_h)} outward expansion` },
      { name: "bearing alignment",     value: alignPos,   hint: `${v.spread_bearing_deg.toFixed(0)}° → cos=${v.alignment_cos.toFixed(2)} toward evac` },
      { name: "fire size amplifier",   value: areaScale,  hint: `${v.area_first_ha.toFixed(0)} ha initial — larger fire, broader front` },
    ],
    summary: {
      ok:    "Low spread pressure — radial expansion not oriented toward evacuation zones.",
      watch: "Moderate spread pressure — fire expanding with partial orientation toward zones.",
      alarm: "High spread pressure — large fire expanding rapidly toward evac zones.",
    }[level],
  };
}


// ─── 3. CONTAINMENT DIFFICULTY ────────────────────────────────────────────────
// Question: How hard will this fire be to suppress and hold?
// Primary axis: Growth rate + model reach probability, amplified by fire size and proximity.

export function scenarioContainmentDifficulty(fgVals, rsVals, rhVals) {
  // rhVals carries reach/probability fields (prob_24h etc.)
  const v          = { ...fgVals, ...rsVals, ...rhVals };
  const sizeScore   = clamp01(Math.log1p(v.area_first_ha) / R("log1p_area_first").max);
  const growthScore = clamp01(Math.sqrt(v.area_growth_rate_ha_per_h / R("area_growth_rate_ha_per_h").max));
  const proxScore   = clamp01(1 - Math.pow(v.dist_min_ci_0_5h / R("dist_min_ci_0_5h").max, 0.5));
  const probScore  = clamp01(v.prob_24h);
  const combined   = clamp01(0.20 * sizeScore + 0.35 * growthScore + 0.15 * proxScore + 0.30 * probScore);
  const level      = alarmLevel(combined);

  return {
    title: "Containment difficulty",
    subtitle: "fire size + growth rate + proximity + reach probability → overall suppression challenge",
    sources: ["area_first_ha", "area_growth_rate_ha_per_h", "dist_min_ci_0_5h", "prob_24h"],
    accentColor: "#5B8DD9",
    overallScore: combined,
    signals: [
      { name: "fire size",      value: sizeScore,   hint: `${v.area_first_ha.toFixed(0)} ha — ${v.area_first_ha > 1000 ? "large" : v.area_first_ha > 100 ? "medium" : "small"} fire` },
      { name: "growth rate",    value: growthScore, hint: `${v.area_growth_rate_ha_per_h.toFixed(2)} ha/h expansion` },
      { name: "zone proximity", value: proxScore,   hint: `${fmtKm(v.dist_min_ci_0_5h)} to nearest evac zone` },
      { name: "24h reach prob", value: probScore,   hint: `${Math.round(v.prob_24h * 100)}% probability within 24h` },
    ],
    summary: {
      ok:    "Containment difficulty is manageable — fire is not large or fast-growing.",
      watch: "Elevated containment difficulty — fire size and reach probability require active resources.",
      alarm: "Very high containment difficulty — large, fast-growing fire with high reach probability.",
    }[level],
  };
}


// ─── 4. TRAJECTORY CONFIDENCE ─────────────────────────────────────────────────
// Question: How reliable is the signal that this fire is closing in on zones?
// Primary axis: R² fit quality × closing slope × acceleration, gated by data density.

export function scenarioTrajectoryConfidence(rsVals, tcVals) {
  const v        = { ...rsVals, ...tcVals };
  const r2       = clamp01(v.dist_fit_r2_0_5h);
  const slopeNeg = clamp01(-v.dist_slope_ci_0_5h / Math.abs(R("dist_slope_ci_0_5h").min));
  const accelNeg = clamp01(-v.dist_accel_m_per_h2 / Math.abs(R("dist_accel_m_per_h2").min));
  const dataOk   = clamp01((v.num_perimeters_0_5h - 1) / (R("num_perimeters_0_5h").max - 1));
  const combined = clamp01(r2 * 0.35 + slopeNeg * 0.35 + accelNeg * 0.15 + dataOk * 0.15);
  const level    = alarmLevel(combined);

  return {
    title: "Trajectory confidence",
    subtitle: "R² fit quality × closing slope × acceleration → how reliably fire is approaching evac zones",
    sources: ["dist_fit_r2_0_5h", "dist_slope_ci_0_5h", "dist_accel_m_per_h2", "num_perimeters_0_5h"],
    accentColor: "#1D9E75",
    overallScore: combined,
    signals: [
      { name: "fit quality (R²)",     value: r2,       hint: `R² = ${v.dist_fit_r2_0_5h.toFixed(2)} — ${r2 > 0.7 ? "reliable trend" : r2 > 0.3 ? "moderate fit" : "noisy / unreliable"}` },
      { name: "closing slope",        value: slopeNeg, hint: `slope = ${v.dist_slope_ci_0_5h.toFixed(1)} m/h — ${v.dist_slope_ci_0_5h < 0 ? "approaching" : "retreating or stable"}` },
      { name: "closing acceleration", value: accelNeg, hint: `accel = ${v.dist_accel_m_per_h2.toFixed(2)} m/h² — ${v.dist_accel_m_per_h2 < 0 ? "speeding up toward zones" : "slowing or stable"}` },
    ],
    summary: {
      ok:    "Trajectory data does not show a reliable closing trend — approach signals are weak or noisy.",
      watch: "Some closing trend detected but fit quality is moderate — monitor for strengthening signal.",
      alarm: "High-confidence closing trajectory — fire is consistently accelerating toward evac zones.",
    }[level],
  };
}


// ─── 5. SURGE / BLOWUP RISK ───────────────────────────────────────────────────
// Question: Is this fire entering a blowup or runaway phase right now?
// Primary axis: Area growth acceleration + closing acceleration + centroid speed,
//               confidence-discounted by data quality.

export function scenarioSurgeRisk(ckVals, fgVals, rsVals, tcVals) {
  const v            = { ...ckVals, ...fgVals, ...rsVals, ...tcVals };
  const growthAccel  = clamp01(Math.sqrt(v.area_growth_rate_ha_per_h / R("area_growth_rate_ha_per_h").max));
  const closingAccel = clamp01(-v.dist_accel_m_per_h2 / Math.abs(R("dist_accel_m_per_h2").min));
  const speedUp      = clamp01(v.centroid_speed_m_per_h / R("centroid_speed_m_per_h").max);
  const dataConf     = v.low_temporal_resolution_0_5h === 1 ? 0.5 : 1.0;
  const surgeSignal  = clamp01((growthAccel * 0.4 + closingAccel * 0.4 + speedUp * 0.2) * dataConf);
  const level        = alarmLevel(surgeSignal);

  return {
    title: "Surge / blowup risk",
    subtitle: "area growth acceleration + closing acceleration + centroid speed → fire entering blowup phase",
    sources: ["area_growth_rate_ha_per_h", "dist_accel_m_per_h2", "centroid_speed_m_per_h", "low_temporal_resolution_0_5h"],
    accentColor: "#D85A30",
    overallScore: surgeSignal,
    signals: [
      { name: "area growth accel", value: growthAccel,  hint: `${v.area_growth_rate_ha_per_h.toFixed(2)} ha/h — ${growthAccel > 0.5 ? "rapid expansion" : "moderate"}` },
      { name: "closing accel",     value: closingAccel, hint: `accel = ${v.dist_accel_m_per_h2.toFixed(2)} m/h² — ${v.dist_accel_m_per_h2 < 0 ? "speeding toward zones" : "not accelerating"}` },
      { name: "centroid speed",    value: speedUp,      hint: `${fmtSpeed(v.centroid_speed_m_per_h)} — ${speedUp > 0.4 ? "fast-moving centroid" : "slow or stationary"}` },
    ],
    summary: {
      ok:    "No surge signals — fire behavior appears stable, not entering blowup phase.",
      watch: "Elevated surge indicators — fire is accelerating in size or approach. Watch closely.",
      alarm: `Critical surge risk — fire showing blowup behavior.${v.low_temporal_resolution_0_5h === 1 ? " Note: sparse data reduces detection confidence." : ""}`,
    }[level],
  };
}


// ─── 6. DATA SPARSITY RISK ────────────────────────────────────────────────────
// Question: Can we trust the other alarm signals, or is the data too thin?
// Primary axis: Temporal coverage flags + R² noise + perimeter count + observation span.
// Note: High score here means OTHER signals should be treated with caution.

export function scenarioDataSparseRisk(rsVals, tcVals) {
  const v         = { ...rsVals, ...tcVals };
  const lowRes    = v.low_temporal_resolution_0_5h === 1 ? 1.0 : 0.0;
  const r2Low     = clamp01(1 - v.dist_fit_r2_0_5h);
  const fewPeri   = clamp01(1 - (v.num_perimeters_0_5h - 1) / (R("num_perimeters_0_5h").max - 1));
  const shortSpan = clamp01(1 - v.dt_first_last_0_5h / R("dt_first_last_0_5h").max);
  const sparsity  = clamp01(0.35 * lowRes + 0.25 * r2Low + 0.25 * fewPeri + 0.15 * shortSpan);
  const level     = alarmLevel(sparsity);

  return {
    title: "Data sparsity risk",
    subtitle: "temporal coverage + fit quality + perimeter count → reliability of all other alarm signals",
    sources: ["low_temporal_resolution_0_5h", "dist_fit_r2_0_5h", "num_perimeters_0_5h", "dt_first_last_0_5h"],
    accentColor: "#888780",
    overallScore: sparsity,
    signals: [
      { name: "low-res flag",     value: lowRes,   hint: `low_temporal_resolution = ${v.low_temporal_resolution_0_5h} — ${lowRes === 1 ? "flagged: dt < 0.5h or single perimeter" : "not flagged"}` },
      { name: "trajectory noise", value: r2Low,    hint: `R² = ${v.dist_fit_r2_0_5h.toFixed(2)} — ${v.dist_fit_r2_0_5h < 0.3 ? "very noisy" : v.dist_fit_r2_0_5h < 0.7 ? "moderate" : "reliable"}` },
      { name: "data density",     value: fewPeri,  hint: `${v.num_perimeters_0_5h} perimeters over ${v.dt_first_last_0_5h.toFixed(1)}h — ${v.num_perimeters_0_5h < 3 ? "critically sparse" : v.num_perimeters_0_5h < 6 ? "sparse" : "adequate"}` },
    ],
    summary: {
      ok:    "Data quality is adequate — alarm signals across all modules are reliable.",
      watch: "Moderate data sparsity — treat other alarm signals with some caution.",
      alarm: "High data sparsity — temporal coverage is too thin to trust trajectory and approach signals. Verify with additional sources.",
    }[level],
  };
}


// ─── 7. FLANKING THREAT ───────────────────────────────────────────────────────
// Question: Is the fire threatening zones off its main axis, not just straight ahead?
// Primary axis: Lateral drift × radial expansion, amplified by fire size.

export function scenarioFlankingThreat(ckVals, fgVals) {
  const v = { ...ckVals, ...fgVals };

  // cross_track ceiling: use the larger absolute bound of the signed range
  const crossCeil   = Math.max(Math.abs(R("cross_track_component").min), R("cross_track_component").max);
  const crossTrack  = clamp01(Math.abs(v.cross_track_component) / crossCeil);
  const frontWidth  = clamp01(v.radial_growth_rate_m_per_h / R("radial_growth_rate_m_per_h").max);
  const sizeAmplify = clamp01(Math.log1p(v.area_first_ha) / R("log1p_area_first").max);
  // Size is amplifier, not additive: small fires have limited flank reach regardless
  const combined    = clamp01((crossTrack * 0.50 + frontWidth * 0.50) * (0.4 + 0.6 * sizeAmplify));
  const level       = alarmLevel(combined);
  const driftDir    = v.cross_track_component > 0 ? "right flank" : "left flank";

  return {
    title: "Flanking threat",
    subtitle: "lateral drift × radial expansion × fire size → threat from fire flanks, not just head",
    sources: ["cross_track_component", "radial_growth_rate_m_per_h", "area_first_ha"],
    accentColor: "#C4651A",
    overallScore: combined,
    signals: [
      { name: "lateral drift",    value: crossTrack,  hint: `${Math.abs(v.cross_track_component).toFixed(1)} m/h sideways (${driftDir})` },
      { name: "radial expansion", value: frontWidth,  hint: `${fmtSpeed(v.radial_growth_rate_m_per_h)} — widening fire front` },
      { name: "size amplifier",   value: sizeAmplify, hint: `${v.area_first_ha.toFixed(0)} ha — ${sizeAmplify > 0.6 ? "large fire, wide flanks" : sizeAmplify > 0.3 ? "moderate size" : "small fire, flanks limited"}` },
    ],
    summary: {
      ok:    "Lateral spread is limited — flanking threat to evac zones is low.",
      watch: `Moderate lateral drift detected (${driftDir}) — zones off the main axis may be at risk.`,
      alarm: `High flanking threat — wide, fast-expanding fire with significant lateral drift. Zones on ${driftDir} require attention.`,
    }[level],
  };
}


// ─── 8. APPROACH CONSISTENCY ──────────────────────────────────────────────────
// Question: Is the fire closing in steadily, or is the approach signal erratic?
// Primary axis: Closing slope gates the score — R² and front coherence only matter
//               if the fire is actually approaching.

export function scenarioApproachConsistency(rsVals, tcVals) {
  const v = { ...rsVals, ...tcVals };

  const fitQuality     = clamp01(v.dist_fit_r2_0_5h);
  const closingSlope   = clamp01(-v.dist_slope_ci_0_5h / Math.abs(R("dist_slope_ci_0_5h").min));
  const frontCoherence = clamp01(1 - v.dist_std_ci_0_5h / R("dist_std_ci_0_5h").max);

  // closingSlope is the gate: if fire isn't closing, fit quality is irrelevant
  const combined = clamp01(closingSlope * (fitQuality * 0.55 + frontCoherence * 0.45));
  const level    = alarmLevel(combined);

  return {
    title: "Approach consistency",
    subtitle: "trajectory fit quality × closing slope × front coherence → how reliably fire is closing in",
    sources: ["dist_fit_r2_0_5h", "dist_slope_ci_0_5h", "dist_std_ci_0_5h"],
    accentColor: "#2E86AB",
    overallScore: combined,
    signals: [
      { name: "fit quality (R²)",  value: fitQuality,    hint: `R² = ${v.dist_fit_r2_0_5h.toFixed(2)} — ${fitQuality > 0.7 ? "consistent trend" : fitQuality > 0.3 ? "moderate fit" : "noisy / erratic"}` },
      { name: "closing slope",     value: closingSlope,  hint: `${v.dist_slope_ci_0_5h.toFixed(1)} m/h — ${v.dist_slope_ci_0_5h < 0 ? "approaching" : "retreating or stable"}` },
      { name: "front coherence",   value: frontCoherence, hint: `dist_std = ${v.dist_std_ci_0_5h.toFixed(0)} m — ${v.dist_std_ci_0_5h < 300 ? "coherent front" : v.dist_std_ci_0_5h < 600 ? "moderate spread" : "fragmented / multi-front"}` },
    ],
    summary: {
      ok:    "No consistent closing trend — approach signals are weak, noisy, or fire is not advancing.",
      watch: "Moderate approach consistency — fire appears to be closing but trend is not fully stable.",
      alarm: "High-confidence sustained approach — fire is closing on evac zones in a consistent, coherent pattern.",
    }[level],
  };
}


// ─── 9. IGNITION TIMING RISK ──────────────────────────────────────────────────
// Question: Did this fire start at a time when natural conditions favor rapid spread?
// Primary axis: Diurnal window (RH trough, peak wind) × growth rate × detection quality.
// Distinct from scenarioOffHoursResponseRisk: this captures environmental/fuel conditions,
// not institutional capacity.

export function scenarioIgnitionTiming(fgVals, tcVals, tmVals) {
  const v    = { ...fgVals, ...tcVals, ...tmVals };
  const hour = v.event_start_hour;

  // Peak diurnal window: RH trough, surface wind peaks, fuel moisture lowest
  const diurnalRisk = (hour >= 11 && hour <= 17) ? 0.9
                    : (hour >= 9  && hour <= 19) ? 0.5
                    : 0.1;

  // Night adds detection/visibility penalty on top of any diurnal concern
  const nightRisk = (hour >= 21 || hour < 5) ? 0.8
                  : (hour >= 18 || hour < 7) ? 0.4
                  : 0.0;

  // Growth rate proxies for "are current fuel/wind conditions actually supporting spread"
  const growthNorm = clamp01(Math.sqrt(v.area_growth_rate_ha_per_h / R("area_growth_rate_ha_per_h").max));
  const detectionPenalty = v.low_temporal_resolution_0_5h === 1 ? 0.3 : 0.0;

  // Max of diurnal and night: they are separate concerns, not additive
  const timeRisk = clamp01(Math.max(diurnalRisk, nightRisk));
  const combined = clamp01(timeRisk * 0.40 + growthNorm * 0.45 + detectionPenalty * 0.15);

  const level   = alarmLevel(combined);
  const hourStr = `${String(hour).padStart(2, "0")}:00`;
  const timeCtx = nightRisk >= 0.8   ? "nighttime — low visibility, suppression window limited"
                : nightRisk >= 0.4   ? "late evening / early morning"
                : diurnalRisk >= 0.8 ? "peak diurnal window — worst-case fuel & wind conditions"
                : diurnalRisk >= 0.4 ? "active burn hours"
                : "off-peak hours";

  return {
    title: "Ignition timing risk",
    subtitle: "diurnal window × growth rate × detection quality → environmental conditions at ignition",
    sources: ["event_start_hour", "area_growth_rate_ha_per_h", "low_temporal_resolution_0_5h"],
    accentColor: "#7F77DD",
    overallScore: combined,
    signals: [
      { name: "time-of-day risk",  value: timeRisk,          hint: `${hourStr} — ${timeCtx}` },
      { name: "growth rate",       value: growthNorm,        hint: `${v.area_growth_rate_ha_per_h.toFixed(2)} ha/h — ${growthNorm > 0.6 ? "rapid, window closing fast" : growthNorm > 0.3 ? "moderate expansion" : "slow growth"}` },
      { name: "detection penalty", value: detectionPenalty,  hint: `low_res = ${v.low_temporal_resolution_0_5h} — ${detectionPenalty > 0 ? "sparse data delays detection" : "adequate temporal coverage"}` },
    ],
    summary: {
      ok:    `${hourStr} ignition with slow growth — environmental conditions are not strongly favorable for spread.`,
      watch: `${hourStr} — ${timeCtx}. Moderate growth suggests conditions are supporting spread.`,
      alarm: `${hourStr} — ${timeCtx}. Rapid growth${v.low_temporal_resolution_0_5h === 1 ? " and sparse detection" : ""} — environmental window is critically unfavorable.`,
    }[level],
  };
}


// ─── 10. RELATIVE GROWTH INTENSITY ────────────────────────────────────────────
// Question: Is this fire exploding relative to its starting size?
// Primary axis: Log area ratio (proportional expansion) × radial growth × relative fraction.
// Catches small fires that have already multiplied 5–10× — a signal absolute rates miss.

export function scenarioRelativeGrowthIntensity(fgVals) {
  const v = { ...fgVals };

  // log_area_ratio = log(area_final / area_first); negative if fire shrank (noise/correction)
  const logRatio       = v.log_area_ratio_0_5h ?? 0;
  const relGrowthScore = clamp01(nn(logRatio) / R("log_area_ratio_0_5h").max);

  // Radial growth in absolute meters; ceiling ~2 km for extreme events
  const radialScore    = clamp01(nn(v.radial_growth_m) / R("radial_growth_m").max);

  // area_growth_rel: fraction — 0.5 = +50%, 2.0 = +200%; log-compressed, ceiling at 5×
  const relFracScore   = clamp01(Math.log1p(nn(v.area_growth_rel_0_5h)) / Math.log1p(5));

  const combined = clamp01(relGrowthScore * 0.45 + radialScore * 0.30 + relFracScore * 0.25);
  const level    = alarmLevel(combined);

  const multipleStr = logRatio >= 0
    ? `${Math.exp(logRatio).toFixed(1)}× initial size`
    : "slightly smaller than initial (noise/correction)";

  return {
    title: "Relative growth intensity",
    subtitle: "log area ratio × radial expansion × relative fraction → early burst detection regardless of initial size",
    sources: ["log_area_ratio_0_5h", "radial_growth_m", "area_growth_rel_0_5h"],
    accentColor: "#E05C2A",
    overallScore: combined,
    signals: [
      { name: "log area ratio",    value: relGrowthScore, hint: `log_ratio = ${logRatio.toFixed(2)} — fire is ${multipleStr}` },
      { name: "radial expansion",  value: radialScore,    hint: `${nn(v.radial_growth_m).toFixed(0)} m radius growth` },
      { name: "relative fraction", value: relFracScore,   hint: `+${(nn(v.area_growth_rel_0_5h) * 100).toFixed(0)}% of initial area` },
    ],
    summary: {
      ok:    "Fire size increase is proportionally modest — no early burst detected.",
      watch: "Significant relative growth — fire is expanding faster than its initial size suggests.",
      alarm: "Explosive relative expansion — fire has multiplied several times from initial size. Early blowup signature.",
    }[level],
  };
}


// ─── 11. PROJECTED ADVANCE RISK ───────────────────────────────────────────────
// Question: How much of the safety buffer has already been consumed, and at what rate?
// Primary axis: Buffer consumed ratio — a fire that has eaten 80% of its remaining
//               distance is critical even if absolute advance looks small.

export function scenarioProjectedAdvance(rsVals) {
  const v = { ...rsVals };

  // projected_advance_m = d0 − d5; negative means fire retreated — treat as 0
  const advance      = nn(v.projected_advance_m);
  const advanceScore = clamp01(advance / R("projected_advance_m").max);

  // Buffer ratio: what fraction of the remaining distance has already been consumed
  const bufferRatio  = v.dist_min_ci_0_5h > 0
    ? clamp01(advance / v.dist_min_ci_0_5h)
    : 1.0;

  const speedScore   = clamp01(v.closing_speed_m_per_h / R("closing_speed_m_per_h").max);

  const combined = clamp01(advanceScore * 0.30 + bufferRatio * 0.50 + speedScore * 0.20);
  const level    = alarmLevel(combined);

  const remainKm = fmtKm(v.dist_min_ci_0_5h);
  const advKm    = fmtKm(advance);

  return {
    title: "Projected advance risk",
    subtitle: "observed advance ÷ remaining buffer × closing speed → near-term zone encroachment",
    sources: ["projected_advance_m", "dist_min_ci_0_5h", "closing_speed_m_per_h"],
    accentColor: "#C0392B",
    overallScore: combined,
    signals: [
      { name: "observed advance", value: advanceScore, hint: `${advKm} advance toward zone in obs. window` },
      { name: "buffer consumed",  value: bufferRatio,  hint: `${remainKm} buffer remaining — advance consumed ${(bufferRatio * 100).toFixed(0)}% of buffer` },
      { name: "closing speed",    value: speedScore,   hint: `${fmtSpeed(v.closing_speed_m_per_h)} forward speed` },
    ],
    summary: {
      ok:    `Fire has advanced ${advKm} but ${remainKm} buffer remains — zone encroachment not imminent.`,
      watch: `Fire has consumed a notable portion of the buffer (${remainKm} left). Monitor closely.`,
      alarm: `Critical encroachment — fire has advanced ${advKm}, only ${remainKm} to nearest evac zone.`,
    }[level],
  };
}


// ─── 12. FRONT FRAGMENTATION ──────────────────────────────────────────────────
// Question: Is this a coherent single-front fire, or a chaotic multi-flank event?
// Primary axis: Spread in perimeter-to-zone distances (dist_std) — high σ means
//               different parts of the perimeter are at very different distances from zones,
//               indicating multiple active fronts rather than one contained head.

export function scenarioFrontFragmentation(rsVals, fgVals) {
  const v = { ...rsVals, ...fgVals };

  // aligned with scenarioApproachConsistency
  const distSpread  = clamp01(v.dist_std_ci_0_5h / R("dist_std_ci_0_5h").max);
  const radialWidth = clamp01(v.radial_growth_rate_m_per_h / R("radial_growth_rate_m_per_h").max);
  // Relative growth amplifies: fragmented + fast-growing = worst case
  const growthAmpli = clamp01(Math.log1p(nn(v.area_growth_rel_0_5h)) / Math.log1p(5));

  // Fragmentation is the primary signal; growth amplifies but does not dominate alone
  const combined = clamp01((distSpread * 0.55 + radialWidth * 0.25) * (0.5 + 0.5 * growthAmpli));
  const level    = alarmLevel(combined);

  return {
    title: "Front fragmentation",
    subtitle: "perimeter distance spread × radial width × growth → coherent single front vs. multi-flank chaos",
    sources: ["dist_std_ci_0_5h", "radial_growth_rate_m_per_h", "area_growth_rel_0_5h"],
    accentColor: "#8E44AD",
    overallScore: combined,
    signals: [
      { name: "distance spread (σ)", value: distSpread,  hint: `dist_std = ${v.dist_std_ci_0_5h.toFixed(0)} m — ${v.dist_std_ci_0_5h > 600 ? "highly fragmented" : v.dist_std_ci_0_5h > 300 ? "moderate fragmentation" : "coherent front"}` },
      { name: "radial front width",  value: radialWidth, hint: `${fmtSpeed(v.radial_growth_rate_m_per_h)} radial — widening perimeter` },
      { name: "growth amplifier",    value: growthAmpli, hint: `+${(nn(v.area_growth_rel_0_5h) * 100).toFixed(0)}% relative growth — amplifies multi-flank risk` },
    ],
    summary: {
      ok:    "Fire perimeter shows coherent, single-front behavior — containment geometry is predictable.",
      watch: "Moderate front fragmentation — multiple flanks may be developing. Trajectory signals less reliable.",
      alarm: "High fragmentation — fire advancing on multiple fronts simultaneously. Containment is complex; trajectory signals are unreliable.",
    }[level],
  };
}


// ─── 13. OFF-HOURS RESPONSE RISK ─────────────────────────────────────────────
// Question: Are suppression resources constrained by time-of-day or day-of-week?
// Primary axis: Night hours (aerial ops unavailable) × weekend (reduced agency staffing),
//               urgency-weighted by growth rate.
// Distinct from scenarioIgnitionTiming: this captures institutional/resource capacity,
// not environmental fuel conditions.

export function scenarioOffHoursResponseRisk(fgVals, tcVals, tmVals) {
  const v = { ...fgVals, ...tcVals, ...tmVals };
  const hour = v.event_start_hour;
  const dow  = v.event_start_dayofweek; // 0 = Monday … 6 = Sunday

  // Nighttime penalty: aerial suppression unavailable, visibility low, staffing reduced
  const nightPenalty = (hour >= 22 || hour < 6) ? 1.0
                     : (hour >= 19 || hour < 8) ? 0.5
                     : 0.0;

  // Weekend penalty: reduced agency staffing, slower mutual aid mobilization
  const isWeekend    = dow >= 5;
  const weekendScore = isWeekend             ? 0.8
                     : (dow === 4 && hour >= 17) ? 0.4  // Friday evening
                     : 0.1;

  // Growth rate: faster fire = the resource gap matters more
  const growthNorm = clamp01(Math.sqrt(v.area_growth_rate_ha_per_h / R("area_growth_rate_ha_per_h").max));
  const timePenalty = clamp01(Math.max(nightPenalty, weekendScore));

  // Interaction term: simultaneous night + weekend + rapid growth is worse than sum of parts
  const combined = clamp01(timePenalty * 0.45 + growthNorm * 0.40 + timePenalty * growthNorm * 0.15);
  const level    = alarmLevel(combined);

  const hourStr  = `${String(hour).padStart(2, "0")}:00`;
  const dayNames = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  const dayStr   = dayNames[dow] ?? "?";
  const timeCtx  = nightPenalty >= 0.8 ? "core night hours — aerial ops unavailable"
                 : nightPenalty >= 0.4 ? "shoulder hours — reduced staffing"
                 : isWeekend           ? "weekend — reduced agency staffing"
                 : "standard operating hours";

  return {
    title: "Off-hours response risk",
    subtitle: "time-of-day × day-of-week × growth rate → suppression resource availability gap",
    sources: ["event_start_hour", "event_start_dayofweek", "area_growth_rate_ha_per_h"],
    accentColor: "#566573",
    overallScore: combined,
    signals: [
      { name: "nighttime penalty", value: nightPenalty,  hint: `${hourStr} — ${nightPenalty >= 0.8 ? "core night, aerial ops limited" : nightPenalty > 0 ? "shoulder hours" : "daytime, full ops"}` },
      { name: "weekend penalty",   value: weekendScore,  hint: `${dayStr} — ${isWeekend ? "weekend: reduced staffing & mutual aid" : "weekday: normal staffing"}` },
      { name: "growth urgency",    value: growthNorm,    hint: `${v.area_growth_rate_ha_per_h.toFixed(2)} ha/h — ${growthNorm > 0.5 ? "rapid: resource gap is critical" : "moderate growth"}` },
    ],
    summary: {
      ok:    `${dayStr} ${hourStr} — ${timeCtx}. Full suppression capacity available.`,
      watch: `${dayStr} ${hourStr} — ${timeCtx}. Immediate response capacity is reduced; monitor growth closely.`,
      alarm: `${dayStr} ${hourStr} — ${timeCtx}. Rapid fire growth with significantly constrained suppression resources.`,
    }[level],
  };
}