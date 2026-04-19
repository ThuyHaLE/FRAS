// src/scenarios/crossModuleScenarios.js
// This file defines composite scenarios that integrate signals from multiple modules 
// to provide higher-level insights about fire behavior and risk. Each scenario function takes relevant input values,
// computes a combined score, and returns a structured object with title, subtitle, sources, signals, 
// and summary messages for different alarm levels.

import { clamp01, alarmLevel } from "./utils/alarmUtils";

export function scenarioETA(ckVals, rsVals) {
  const v = { ...ckVals, ...rsVals };
  const effectiveSpeed = Math.max(v.centroid_speed_m_per_h * Math.max(v.alignment_cos ?? 0, 0), v.closing_speed_m_per_h);
  const etaH       = effectiveSpeed > 0.5 ? v.dist_min_ci_0_5h / effectiveSpeed : null;
  const etaScore   = effectiveSpeed > 0.5 ? clamp01(1 - etaH / 72) : 0;
  const alignScore = clamp01(((v.alignment_cos ?? 0) - 0.2) / 0.8);
  const speedScore = clamp01(v.centroid_speed_m_per_h / 600);
  const combined   = clamp01(etaScore * 0.6 + alignScore * 0.25 + speedScore * 0.15);
  const etaLabel   = etaH === null ? "N/A (stationary)" : etaH < 1 ? `~${Math.round(etaH * 60)} min` : `~${etaH.toFixed(1)} h`;
  const level      = alarmLevel(combined);
  return {
    title: "Time-to-reach estimate",
    subtitle: "centroid speed × bearing alignment → effective approach toward nearest evac zone",
    sources: ["centroid_speed_m_per_h","alignment_cos","dist_min_ci_0_5h","closing_speed_m_per_h"],
    accentColor: "#378ADD",
    overallScore: combined,
    signals: [
      { name: "effective closing speed", value: speedScore,  hint: `${effectiveSpeed.toFixed(1)} m/h (speed × alignment)` },
      { name: "alignment toward zone",   value: alignScore,  hint: `alignment_cos = ${(v.alignment_cos??0).toFixed(2)} → ${(v.alignment_cos??0)>0.7?"direct":(v.alignment_cos??0)>0.3?"oblique":"tangential"}` },
      { name: "ETA estimate",            value: etaScore,    hint: `${etaLabel} at current effective speed` },
    ],
    summary: { ok: `Fire is not approaching evac zones at this time. ETA: ${etaLabel}.`, watch: `Fire may reach zones in ${etaLabel} if current bearing holds.`, alarm: `Rapid approach — ETA ${etaLabel}. Immediate action window.` }[level],
  };
}

export function scenarioSpreadPressure(ckVals, fgVals, dirVals) {
  const v          = { ...ckVals, ...fgVals, ...dirVals };
  const radialNorm = clamp01(v.radial_growth_rate_m_per_h / 360);
  const alignPos   = clamp01(v.alignment_cos);
  const areaScale  = clamp01(Math.log1p(v.area_first_ha) / 9.4);
  const pressure   = clamp01(radialNorm * alignPos * (0.5 + 0.5 * areaScale));
  const level      = alarmLevel(pressure);
  return {
    title: "Directional spread pressure",
    subtitle: "radial expansion rate × bearing alignment × fire size → net push toward evac zones",
    sources: ["radial_growth_rate_m_per_h","alignment_cos","area_first_ha","spread_bearing_deg"],
    accentColor: "#D85A30",
    overallScore: pressure,
    signals: [
      { name: "radial expansion rate", value: radialNorm, hint: `${v.radial_growth_rate_m_per_h.toFixed(1)} m/h outward expansion` },
      { name: "bearing alignment",     value: alignPos,   hint: `${v.spread_bearing_deg.toFixed(0)}° → cos=${v.alignment_cos.toFixed(2)} toward evac` },
      { name: "fire size amplifier",   value: areaScale,  hint: `${v.area_first_ha.toFixed(0)} ha initial — larger fire, broader front` },
    ],
    summary: { ok: "Low spread pressure — radial expansion not oriented toward evacuation zones.", watch: "Moderate spread pressure — fire expanding with partial orientation toward zones.", alarm: "High spread pressure — large fire expanding rapidly toward evac zones." }[level],
  };
}

export function scenarioNightScale(fgVals, tcVals, tmVals) {
  const v           = { ...fgVals, ...tcVals, ...tmVals };
  const hour        = v.event_start_hour;
  const isNight     = hour >= 20 || hour < 6;
  const isPeak      = hour >= 11 && hour < 17;
  const timeRisk    = isNight ? 0.8 : isPeak ? 0.4 : 0.2;
  const growthNorm  = clamp01(Math.sqrt(v.area_growth_rate_ha_per_h / 525));
  const coverageOk  = v.num_perimeters_0_5h >= 5 && v.dt_first_last_0_5h >= 1;
  const detRisk     = coverageOk ? 0.3 : 0.7;
  const combined    = clamp01(timeRisk * 0.4 + growthNorm * 0.35 + detRisk * 0.25);
  const hourStr     = String(hour).padStart(2, "0") + ":00";
  const level       = alarmLevel(combined);
  return {
    title: "Detection & response window",
    subtitle: "start time + growth rate + temporal coverage → how much reaction time remains",
    sources: ["event_start_hour","area_growth_rate_ha_per_h","num_perimeters_0_5h","dt_first_last_0_5h"],
    accentColor: "#7F77DD",
    overallScore: combined,
    signals: [
      { name: "time-of-day risk",  value: timeRisk,        hint: `${hourStr} — ${isNight ? "nighttime: low visibility" : isPeak ? "peak heat hours" : "daytime off-peak"}` },
      { name: "growth rate",       value: growthNorm,       hint: `${v.area_growth_rate_ha_per_h.toFixed(2)} ha/h — ${growthNorm > 0.5 ? "rapid" : "moderate"} expansion` },
      { name: "detection quality", value: 1 - detRisk,     hint: `${v.num_perimeters_0_5h} perimeters / ${v.dt_first_last_0_5h.toFixed(1)}h — ${coverageOk ? "adequate" : "sparse data"}` },
    ],
    summary: { ok: `Fire started at ${hourStr} with adequate coverage — reaction window is reasonable.`, watch: `${isNight ? "Nighttime" : "Peak-hour"} start reduces early detection reliability. Monitor growth closely.`, alarm: `Nighttime ignition with rapid growth and sparse data — response window critically narrow.` }[level],
  };
}

export function scenarioContainmentDifficulty(fgVals, rsVals, rhVals) {
  const v          = { ...fgVals, ...rsVals, ...rhVals };
  const sizeScore  = clamp01(Math.log1p(v.area_first_ha) / 9.4);
  const growthScore = clamp01(Math.sqrt(v.area_growth_rate_ha_per_h / 525));
  const proxScore  = clamp01(1 - Math.pow(v.dist_min_ci_0_5h / 800000, 0.5));
  const probScore  = clamp01(v.prob_h24);
  const combined   = clamp01(0.3 * sizeScore + 0.25 * growthScore + 0.2 * proxScore + 0.25 * probScore);
  const level      = alarmLevel(combined);
  return {
    title: "Containment difficulty",
    subtitle: "fire size + growth rate + proximity + reach probability → overall suppression challenge",
    sources: ["area_first_ha","area_growth_rate_ha_per_h","dist_min_ci_0_5h","prob_h24"],
    accentColor: "#5B8DD9",
    overallScore: combined,
    signals: [
      { name: "fire size",      value: sizeScore,   hint: `${v.area_first_ha.toFixed(0)} ha — ${v.area_first_ha > 1000 ? "large" : v.area_first_ha > 100 ? "medium" : "small"} fire` },
      { name: "growth rate",    value: growthScore, hint: `${v.area_growth_rate_ha_per_h.toFixed(2)} ha/h expansion` },
      { name: "zone proximity", value: proxScore,   hint: `${(v.dist_min_ci_0_5h / 1000).toFixed(1)} km to nearest evac zone` },
      { name: "24h reach prob", value: probScore,   hint: `${Math.round(v.prob_h24 * 100)}% probability within 24h` },
    ],
    summary: { ok: "Containment difficulty is manageable — fire is not large or fast-growing.", watch: "Elevated containment difficulty — fire size and reach probability require active resources.", alarm: "Very high containment difficulty — large, fast-growing fire with high reach probability." }[level],
  };
}

export function scenarioTrajectoryConfidence(rsVals, tcVals) {
  const v        = { ...rsVals, ...tcVals };
  const r2       = clamp01(v.dist_fit_r2_0_5h);
  const slopeNeg = clamp01((-v.dist_slope_ci_0_5h) / 600);
  const accelNeg = clamp01((-v.dist_accel_m_per_h2) / 120);
  const dataOk   = clamp01((v.num_perimeters_0_5h - 1) / 19);
  const combined = clamp01(r2 * 0.4 * slopeNeg * accelNeg + r2 * 0.3 * slopeNeg + dataOk * 0.3);
  const level    = alarmLevel(combined);
  return {
    title: "Trajectory confidence",
    subtitle: "R² fit quality × closing slope × acceleration → how reliably fire is approaching evac zones",
    sources: ["dist_fit_r2_0_5h","dist_slope_ci_0_5h","dist_accel_m_per_h2","num_perimeters_0_5h"],
    accentColor: "#1D9E75",
    overallScore: combined,
    signals: [
      { name: "fit quality (R²)",     value: r2,       hint: `R² = ${v.dist_fit_r2_0_5h.toFixed(2)} — ${r2 > 0.7 ? "reliable trend" : r2 > 0.3 ? "moderate fit" : "noisy / unreliable"}` },
      { name: "closing slope",        value: slopeNeg, hint: `slope = ${v.dist_slope_ci_0_5h.toFixed(1)} m/h — ${v.dist_slope_ci_0_5h < 0 ? "approaching" : "retreating or stable"}` },
      { name: "closing acceleration", value: accelNeg, hint: `accel = ${v.dist_accel_m_per_h2.toFixed(2)} m/h² — ${v.dist_accel_m_per_h2 < 0 ? "speeding up toward zones" : "slowing or stable"}` },
    ],
    summary: { ok: "Trajectory data does not show a reliable closing trend — approach signals are weak or noisy.", watch: "Some closing trend detected but fit quality is moderate — monitor for strengthening signal.", alarm: "High-confidence closing trajectory — fire is consistently accelerating toward evac zones." }[level],
  };
}

export function scenarioSurgeRisk(ckVals, fgVals, rsVals, tcVals) {
  const v           = { ...ckVals, ...fgVals, ...rsVals, ...tcVals };
  const growthAccel = clamp01(Math.sqrt(v.area_growth_rate_ha_per_h / 525));
  const closingAccel = clamp01((-v.dist_accel_m_per_h2) / 120);
  const speedUp     = clamp01(v.centroid_speed_m_per_h / 600);
  const dataConf    = v.low_temporal_resolution_0_5h === 1 ? 0.5 : 1.0;
  const surgeSignal = clamp01((growthAccel * 0.4 + closingAccel * 0.4 + speedUp * 0.2) * dataConf);
  const level       = alarmLevel(surgeSignal);
  return {
    title: "Surge / blowup risk",
    subtitle: "area growth acceleration + closing acceleration + centroid speed → fire entering blowup phase",
    sources: ["area_growth_rate_ha_per_h","dist_accel_m_per_h2","centroid_speed_m_per_h","low_temporal_resolution_0_5h"],
    accentColor: "#D85A30",
    overallScore: surgeSignal,
    signals: [
      { name: "area growth accel", value: growthAccel,  hint: `${v.area_growth_rate_ha_per_h.toFixed(2)} ha/h — ${growthAccel > 0.5 ? "rapid expansion" : "moderate"}` },
      { name: "closing accel",     value: closingAccel, hint: `accel = ${v.dist_accel_m_per_h2.toFixed(2)} m/h² — ${v.dist_accel_m_per_h2 < 0 ? "speeding toward zones" : "not accelerating"}` },
      { name: "centroid speed",    value: speedUp,      hint: `${v.centroid_speed_m_per_h.toFixed(1)} m/h — ${speedUp > 0.4 ? "fast-moving centroid" : "slow or stationary"}` },
    ],
    summary: { ok: "No surge signals — fire behavior appears stable, not entering blowup phase.", watch: "Elevated surge indicators — fire is accelerating in size or approach. Watch closely.", alarm: `Critical surge risk — fire showing blowup behavior.${v.low_temporal_resolution_0_5h === 1 ? " Note: sparse data reduces detection confidence." : ""}` }[level],
  };
}

export function scenarioSeasonalVulnerability(fgVals, rsVals, tmVals) {
  const v          = { ...fgVals, ...rsVals, ...tmVals };
  const month      = v.event_start_month;
  const seasonRisk = (month >= 6 && month <= 9) ? 1.0 : (month === 5 || month === 10) ? 0.6 : (month === 4 || month === 11) ? 0.3 : 0.1;
  const sizeScore  = clamp01(Math.log1p(v.area_first_ha) / 9.4);
  const growthScore = clamp01(Math.sqrt(v.area_growth_rate_ha_per_h / 525));
  const proxScore  = clamp01(1 - Math.pow(v.dist_min_ci_0_5h / 800000, 0.5));
  const baseRisk   = clamp01(0.35 * sizeScore + 0.35 * growthScore + 0.30 * proxScore);
  const combined   = clamp01(baseRisk * (0.4 + 0.6 * seasonRisk));
  const monthName  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][month - 1] ?? "?";
  const level      = alarmLevel(combined);
  return {
    title: "Seasonal vulnerability",
    subtitle: "fire season month × fire size × growth rate × proximity → environmental risk amplification",
    sources: ["event_start_month","area_first_ha","area_growth_rate_ha_per_h","dist_min_ci_0_5h"],
    accentColor: "#BA7517",
    overallScore: combined,
    signals: [
      { name: "season risk",    value: seasonRisk, hint: `${monthName} — ${seasonRisk >= 1 ? "peak fire season" : seasonRisk >= 0.6 ? "shoulder season" : "off-season"}` },
      { name: "fire baseline",  value: baseRisk,   hint: `size ${sizeScore.toFixed(2)} + growth ${growthScore.toFixed(2)} + proximity ${proxScore.toFixed(2)}` },
      { name: "amplified risk", value: combined,   hint: `baseline × season factor (${(0.4 + 0.6 * seasonRisk).toFixed(2)})` },
    ],
    summary: { ok: `${monthName} is off-season — current fire behavior carries lower ambient risk.`, watch: `${monthName} is shoulder season — fire conditions are moderately elevated by seasonal context.`, alarm: `${monthName} is peak fire season — all risk signals are significantly amplified by environmental conditions.` }[level],
  };
}

export function scenarioDataSparseRisk(rsVals, tcVals) {
  const v         = { ...rsVals, ...tcVals };
  const lowRes    = v.low_temporal_resolution_0_5h === 1 ? 1.0 : 0.0;
  const r2Low     = clamp01(1 - v.dist_fit_r2_0_5h);
  const fewPeri   = clamp01(1 - (v.num_perimeters_0_5h - 1) / 19);
  const shortSpan = clamp01(1 - v.dt_first_last_0_5h / 5);
  const sparsity  = clamp01(0.35 * lowRes + 0.25 * r2Low + 0.25 * fewPeri + 0.15 * shortSpan);
  const level     = alarmLevel(sparsity);
  return {
    title: "Data sparsity risk",
    subtitle: "temporal coverage + fit quality + perimeter count → reliability of all other alarm signals",
    sources: ["low_temporal_resolution_0_5h","dist_fit_r2_0_5h","num_perimeters_0_5h","dt_first_last_0_5h"],
    accentColor: "#888780",
    overallScore: sparsity,
    signals: [
      { name: "low-res flag",     value: lowRes,   hint: `low_temporal_resolution = ${v.low_temporal_resolution_0_5h} — ${lowRes === 1 ? "flagged: dt < 0.5h or single perimeter" : "not flagged"}` },
      { name: "trajectory noise", value: r2Low,    hint: `R² = ${v.dist_fit_r2_0_5h.toFixed(2)} — ${v.dist_fit_r2_0_5h < 0.3 ? "very noisy" : v.dist_fit_r2_0_5h < 0.7 ? "moderate" : "reliable"}` },
      { name: "data density",     value: fewPeri,  hint: `${v.num_perimeters_0_5h} perimeters over ${v.dt_first_last_0_5h.toFixed(1)}h — ${v.num_perimeters_0_5h < 3 ? "critically sparse" : v.num_perimeters_0_5h < 6 ? "sparse" : "adequate"}` },
    ],
    summary: { ok: "Data quality is adequate — alarm signals across all modules are reliable.", watch: "Moderate data sparsity — treat other alarm signals with some caution.", alarm: "High data sparsity — temporal coverage is too thin to trust trajectory and approach signals. Verify with additional sources." }[level],
  };
}