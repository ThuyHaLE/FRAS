// dashboard/src/alarms/tabAlarms.jsx
// Tab for cross-module alarms and scenarios 
// Each scenario combines multiple features across modules to assess specific risk factors 
// or forecast outcomes related to fire behavior and impact.
// Scenarios are designed to be interpretable and actionable for emergency managers, 
// providing clear signals and summaries that can inform decision-making during fire events.

import { clamp01, alarmLevel } from "../utils/alarmUtils";
import { CrossModuleAlarmCard } from "../components/shared/CrossModuleAlarmCard";
import { FEATURE_RANGES, R } from "../constants/featureRanges";
import { scenarioETA, scenarioSpreadPressure, scenarioNightScale,
         scenarioContainmentDifficulty, scenarioTrajectoryConfidence,
         scenarioSurgeRisk, scenarioSeasonalVulnerability,
         scenarioDataSparseRisk } from "../scenarios/crossModuleScenarios";
         
const R = (id) => FEATURE_RANGES[id];

export function useCkAlarm(vals) {
  const displacement = clamp01(vals.centroid_displacement_m / R("centroid_displacement_m").max);
  const speed        = clamp01(vals.centroid_speed_m_per_h / R("centroid_speed_m_per_h").max);
  const mobility     = clamp01(0.3 * displacement + 0.3 * speed + 0.4 * displacement * speed);
  const level        = alarmLevel(mobility);
  return {
    overallScore: mobility,
    signals: [
      { name: "displacement", value: displacement, hint: `${Math.round(vals.centroid_displacement_m)} m total centroid shift` },
      { name: "speed",        value: speed,        hint: `${Math.round(vals.centroid_speed_m_per_h)} m/h centroid speed` },
      { name: "mobility",     value: mobility,     hint: "Combined mobility score" },
    ],
    summary: {
      ok:    "Fire centroid is stationary — no directional threat from movement yet.",
      watch: "Centroid is moving — monitor spread direction relative to evac zones.",
      alarm: "Rapid centroid movement detected — fire is spreading across terrain.",
    }[level],
  };
}

export function useFgAlarm(vals) {
  const size      = clamp01(vals.log1p_area_first / R("log1p_area_first").max);
  const rate      = clamp01(Math.sqrt(vals.area_growth_rate_ha_per_h / R("area_growth_rate_ha_per_h").max));
  const rel       = clamp01(vals.log_area_ratio_0_5h / R("log_area_ratio_0_5h").max);
  const intensity = clamp01(0.3 * size + 0.7 * rate * rel);
  const level     = alarmLevel(intensity);
  return {
    overallScore: intensity,
    signals: [
      { name: "initial size", value: size,      hint: `${vals.area_first_ha.toFixed(0)} ha baseline (log-scaled)` },
      { name: "growth rate",  value: rate,      hint: `${vals.area_growth_rate_ha_per_h.toFixed(1)} ha/h (sqrt-scaled)` },
      { name: "intensity",    value: intensity, hint: "0.3×size + 0.7×(rate × rel)" },
    ],
    summary: {
      ok:    "Fire size and growth rate are within manageable range.",
      watch: "Fire is large or expanding — growth signals warrant close monitoring.",
      alarm: "Rapid fire expansion detected — containment window is narrowing.",
    }[level],
  };
}

export function useDirAlarm(vals) {
  const directness = clamp01((vals.alignment_cos - 0.2) / 0.8);
  const crossMax   = Math.max(Math.abs(R("cross_track_component").min), R("cross_track_component").max);
  const focus      = clamp01(1 - Math.pow(Math.abs(vals.cross_track_component) / crossMax, 0.6));
  const approach   = clamp01((Math.max(vals.along_track_speed, 0) / R("along_track_speed").max - 0.1) / 0.9);
  const alarm      = directness < 0.2 ? 0 : directness * focus * approach;
  const level      = alarmLevel(alarm);
  return {
    overallScore: alarm,
    signals: [
      { name: "alignment",         value: directness, hint: `alignment_cos = ${vals.alignment_cos.toFixed(2)} → ${vals.alignment_cos > 0.7 ? "direct" : vals.alignment_cos > 0.3 ? "oblique" : "neutral"}` },
      { name: "focus (1−drift)",   value: focus,      hint: `cross_track = ${vals.cross_track_component.toFixed(0)} m/h` },
      { name: "directional alarm", value: alarm,      hint: "alignment × focus × approach" },
    ],
    summary: {
      ok:    "Direction is neutral — fire is not oriented toward evacuation zones.",
      watch: "Fire shows partial alignment toward evac zones — track bearing changes.",
      alarm: "Fire is heading directly toward evacuation zones at speed.",
    }[level],
  };
}

export function useRsAlarm(vals) {
  const proximity  = clamp01(1 - Math.pow(vals.dist_min_ci_0_5h / R("dist_min_ci_0_5h").max, 0.5));
  const closing    = clamp01(Math.max(vals.closing_speed_m_per_h, 0) / R("closing_speed_m_per_h").max);
  const advance    = clamp01(Math.max(vals.projected_advance_m, 0) / R("projected_advance_m").max);
  const conf       = clamp01(vals.dist_fit_r2_0_5h);
  const alarm      = proximity * closing * advance * (0.3 + 0.7 * conf);
  const distKm     = (vals.dist_min_ci_0_5h / 1000).toFixed(1);
  const level      = alarmLevel(alarm);
  return {
    overallScore: alarm,
    signals: [
      { name: "proximity",       value: proximity, hint: `${distKm} km to nearest zone (inverted)` },
      { name: "closing speed",   value: closing,   hint: `${Math.max(vals.closing_speed_m_per_h, 0).toFixed(0)} m/h approach rate` },
      { name: "advance",         value: advance,   hint: `${Math.max(vals.projected_advance_m, 0).toFixed(0)} m projected` },
      { name: "proximity alarm", value: alarm,     hint: `proximity × closing × advance × conf(${conf.toFixed(2)})` },
    ],
    summary: {
      ok:    `Fire remains at a safe distance (${distKm} km) with no consistent approach detected.`,
      watch: `Fire is ${distKm} km away and showing approach signals — monitor closely.`,
      alarm: `Fire is close (${distKm} km) and actively advancing toward evac zones.`,
    }[level],
  };
}
         
export function CrossModuleAlarms({ ckVals, fgVals, dirVals, rsVals, tcVals, tmVals, rhVals }) {
  const scenarios = [
    scenarioETA(ckVals, { ...dirVals, ...rsVals }),
    scenarioSpreadPressure(ckVals, fgVals, dirVals),
    scenarioNightScale(fgVals, tcVals, tmVals),
    scenarioContainmentDifficulty(fgVals, rsVals, rhVals),
    scenarioTrajectoryConfidence(rsVals, tcVals),
    scenarioSurgeRisk(ckVals, fgVals, rsVals, tcVals),
    scenarioSeasonalVulnerability(fgVals, rsVals, tmVals),
    scenarioDataSparseRisk(rsVals, tcVals),
  ];
  return (
    <div style={{ marginBottom: 14 }}>
      {scenarios.map((sc) => <CrossModuleAlarmCard key={sc.title} scenario={sc} />)}
    </div>
  );
}