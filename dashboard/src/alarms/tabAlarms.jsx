// dashboard/src/alarms/tabAlarms.jsx
// Tab for cross-module alarms and scenarios 
// Each scenario combines multiple features across modules to assess specific risk factors 
// or forecast outcomes related to fire behavior and impact.
// Scenarios are designed to be interpretable and actionable for emergency managers, 
// providing clear signals and summaries that can inform decision-making during fire events.
//
// MODULE-LEVEL ALARM HOOKS
// All normalizations use R(id).max (and R(id).min where the feature has a meaningful
// negative range) sourced from FEATURE_RANGES, so ceilings stay in sync when ranges
// are updated from real data — no hardcoded magic numbers here.

import { clamp01, alarmLevel } from "../utils/alarmUtils";
import { CrossModuleAlarmCard } from "../components/shared/CrossModuleAlarmCard";
import { FEATURE_RANGES, R } from "../constants/featureRanges";
import {
  scenarioETA,
  scenarioSpreadPressure,
  scenarioContainmentDifficulty,
  scenarioTrajectoryConfidence,
  scenarioSurgeRisk,
  scenarioDataSparseRisk,
  scenarioFlankingThreat,
  scenarioApproachConsistency,
  scenarioIgnitionTiming,
  scenarioRelativeGrowthIntensity,
  scenarioProjectedAdvance,
  scenarioFrontFragmentation,
  scenarioOffHoursResponseRisk,
} from "../scenarios/crossModuleScenarios";

// ─── MODULE-LEVEL ALARM HOOKS ────────────────────────────────────────────────
//
// Design rules applied consistently across all hooks:
//   1. All normalization ceilings come from R(id).max / R(id).min — never hardcoded
//   2. Features with meaningful negative ranges use min-max scaling, not /max
//   3. No raw product of more than two normalized terms (avoids score collapse)
//   4. Negative feature values that represent "no risk" are clamped to 0 explicitly
//      before use, with a comment explaining why

// ── Centroid Kinematics
// Measures how far and fast the fire centroid has moved.
// Both displacement and speed are non-negative by definition.
export function useCkAlarm(vals) {
  const displacement = clamp01(vals.centroid_displacement_m / R("centroid_displacement_m").max);
  const speed        = clamp01(vals.centroid_speed_m_per_h  / R("centroid_speed_m_per_h").max);

  // Weighted sum + interaction: fire that is both far-displaced AND fast is
  // disproportionately dangerous. Interaction weight kept at 0.20 so it amplifies
  // rather than dominates when both terms are high.
  const mobility = clamp01(0.35 * displacement + 0.45 * speed + 0.20 * displacement * speed);
  const level    = alarmLevel(mobility);

  return {
    overallScore: mobility,
    signals: [
      { name: "displacement", value: displacement, hint: `${Math.round(vals.centroid_displacement_m)} m total centroid shift` },
      { name: "speed",        value: speed,        hint: `${Math.round(vals.centroid_speed_m_per_h)} m/h centroid speed` },
      { name: "mobility",     value: mobility,     hint: "0.35×displacement + 0.45×speed + 0.20×interaction" },
    ],
    summary: {
      ok:    "Fire centroid is stationary — no directional threat from movement yet.",
      watch: "Centroid is moving — monitor spread direction relative to evac zones.",
      alarm: "Rapid centroid movement detected — fire is spreading across terrain.",
    }[level],
  };
}

// ── Fire Growth
// Measures how fast the fire is expanding, both in absolute and relative terms.
// log_area_ratio can be negative when perimeter corrections shrink the measured
// area — treat as zero risk (not a suppression signal).
export function useFgAlarm(vals) {
  const size = clamp01(vals.log1p_area_first / R("log1p_area_first").max);

  // sqrt-scaling matches the toNorm used in FG_FEATURES for this field
  const rate = clamp01(Math.sqrt(vals.area_growth_rate_ha_per_h / R("area_growth_rate_ha_per_h").max));

  // Guard negative values: shrinking fire = 0 relative risk, not a negative score
  const rawRel = Math.max(vals.log_area_ratio_0_5h, 0);
  const rel    = clamp01(rawRel / R("log_area_ratio_0_5h").max);

  // rel acts as amplifier on rate: high absolute rate with no relative expansion
  // (e.g. already-huge fire growing slowly) scores lower than a small fire exploding.
  // size is additive baseline so even slow-growing large fires aren't invisible.
  const intensity = clamp01(0.25 * size + 0.45 * rate + 0.30 * rate * rel);
  const level     = alarmLevel(intensity);

  return {
    overallScore: intensity,
    signals: [
      { name: "initial size", value: size,      hint: `${vals.area_first_ha.toFixed(0)} ha baseline (log-scaled)` },
      { name: "growth rate",  value: rate,      hint: `${vals.area_growth_rate_ha_per_h.toFixed(1)} ha/h (sqrt-scaled)` },
      { name: "rel. burst",   value: rel,       hint: `log_area_ratio = ${vals.log_area_ratio_0_5h.toFixed(2)} — ${rawRel > 0 ? `fire is ${Math.exp(rawRel).toFixed(1)}× initial size` : "no net expansion (noise/correction)"}` },
      { name: "intensity",    value: intensity, hint: "0.25×size + 0.45×rate + 0.30×(rate × rel)" },
    ],
    summary: {
      ok:    "Fire size and growth rate are within manageable range.",
      watch: "Fire is large or expanding — growth signals warrant close monitoring.",
      alarm: "Rapid fire expansion detected — containment window is narrowing.",
    }[level],
  };
}

// ── Directionality
// Measures whether the fire is oriented and moving toward evac zones.
//
// Scoring strategy: avoid triple-product collapse by splitting into two stages:
//   Stage 1 — orientation: is the fire pointed at zones? (directness × focus)
//   Stage 2 — weighted combination with approach speed
//
// along_track_speed has a meaningful negative range (fire retreating) — use min-max.
// cross_track_component is symmetric risk on both sides — use absolute value / max range.
export function useDirAlarm(vals) {
  // alignment_cos ∈ [-1, 1]; threshold 0.15 filters truly tangential movement
  const directness = clamp01((vals.alignment_cos - 0.15) / 0.85);

  // cross_track: risk is symmetric (drift left = drift right in terms of missing zones)
  // use the larger absolute bound as the ceiling for consistent scaling
  const crossCeil = Math.max(Math.abs(R("cross_track_component").min), R("cross_track_component").max);
  const focus     = clamp01(1 - Math.pow(Math.abs(vals.cross_track_component) / crossCeil, 0.6));

  // Stage 1: orientation score — how directly is the fire aimed at zones?
  const orientation = clamp01(directness * 0.60 + focus * 0.40);

  // along_track_speed ∈ [-530, 390]; positive = toward zones, negative = retreating
  // clamp negative to 0 then normalize against the positive ceiling only —
  // retreating fire contributes nothing to approach score
  const approachRaw = Math.max(vals.along_track_speed, 0);
  const approach    = clamp01(approachRaw / R("along_track_speed").max);

  // Stage 2: hard gate on directness — if fire isn't even partially aligned,
  // approach speed is irrelevant (could be moving fast but sideways).
  // orientation and approach blended equally once gate passes.
  const alarm = directness < 0.15 ? 0
              : clamp01(orientation * 0.50 + approach * 0.50);
  const level = alarmLevel(alarm);

  return {
    overallScore: alarm,
    signals: [
      { name: "alignment",       value: directness,  hint: `alignment_cos = ${vals.alignment_cos.toFixed(2)} → ${vals.alignment_cos > 0.7 ? "direct" : vals.alignment_cos > 0.3 ? "oblique" : "neutral"}` },
      { name: "focus (1−drift)", value: focus,       hint: `cross_track = ${vals.cross_track_component.toFixed(0)} m/h` },
      { name: "approach speed",  value: approach,    hint: `along_track = ${approachRaw.toFixed(0)} m/h toward zones` },
      { name: "dir. alarm",      value: alarm,       hint: "orientation(align×focus)×0.5 + approach×0.5" },
    ],
    summary: {
      ok:    "Direction is neutral — fire is not oriented toward evacuation zones.",
      watch: "Fire shows partial alignment toward evac zones — track bearing changes.",
      alarm: "Fire is heading directly toward evacuation zones at speed.",
    }[level],
  };
}

// ── Risk Score (Proximity)
// Measures how close the fire is and how fast it is closing in.
//
// Scoring strategy: avoid quadruple-product collapse by grouping into two composites:
//   proxSignal    — where the fire is relative to zones (distance + advance)
//   approachSignal — how it's moving (closing speed confidence-weighted by R²)
// Then blend the two composites.
//
// closing_speed_m_per_h ∈ [-55, 360]: negative = retreating, clamp to 0.
// projected_advance_m   ∈ [-250, 1800]: negative = retreating, clamp to 0.
export function useRsAlarm(vals) {
  // ── Proximity composite
  // sqrt-scaling so mid-range distances still register meaningfully
  const proximity  = clamp01(1 - Math.pow(vals.dist_min_ci_0_5h / R("dist_min_ci_0_5h").max, 0.5));

  // projected_advance: negative means fire retreated — zero risk contribution
  const advanceRaw = Math.max(vals.projected_advance_m, 0);
  const advance    = clamp01(advanceRaw / R("projected_advance_m").max);

  const proxSignal = clamp01(proximity * 0.65 + advance * 0.35);

  // ── Approach composite
  // closing_speed: use positive side only (retreating = no closing risk)
  const closingRaw = Math.max(vals.closing_speed_m_per_h, 0);
  const closing    = clamp01(closingRaw / R("closing_speed_m_per_h").max);

  // R² as confidence weight on closing signal — noisy trajectory discounts urgency
  // floor at 0.35 so even sparse data produces a non-zero approach signal
  const conf           = clamp01(vals.dist_fit_r2_0_5h);
  const confFactor     = 0.35 + 0.65 * conf;
  const approachSignal = clamp01(closing * confFactor);

  // ── Final blend: proximity sets the stakes, approach sets the urgency
  const alarm  = clamp01(proxSignal * 0.55 + approachSignal * 0.45);
  const distKm = (vals.dist_min_ci_0_5h / 1000).toFixed(1);
  const level  = alarmLevel(alarm);

  return {
    overallScore: alarm,
    signals: [
      { name: "proximity",       value: proximity,     hint: `${distKm} km to nearest zone (sqrt-inverted)` },
      { name: "advance",         value: advance,       hint: `${advanceRaw.toFixed(0)} m projected advance` },
      { name: "closing speed",   value: closing,       hint: `${closingRaw.toFixed(0)} m/h approach rate` },
      { name: "trajectory conf", value: conf,          hint: `R² = ${vals.dist_fit_r2_0_5h.toFixed(2)} — ${conf > 0.7 ? "reliable" : conf > 0.3 ? "moderate" : "noisy"}` },
      { name: "proximity alarm", value: alarm,         hint: `proxSignal(${proxSignal.toFixed(2)})×0.55 + approachSignal(${approachSignal.toFixed(2)})×0.45` },
    ],
    summary: {
      ok:    `Fire remains at a safe distance (${distKm} km) with no consistent approach detected.`,
      watch: `Fire is ${distKm} km away and showing approach signals — monitor closely.`,
      alarm: `Fire is close (${distKm} km) and actively advancing toward evac zones.`,
    }[level],
  };
}

// ─── CROSS-MODULE SCENARIO PANEL ─────────────────────────────────────────────
//
// Scenario ordering rationale:
//   1. Data quality first  — DataSparseRisk primes the reader to weight other signals appropriately
//   2. Ignition context    — timing scenarios set the "when" before the "what"
//   3. Fire behavior       — growth and spread shape
//   4. Directional threat  — approach, trajectory, flanking
//   5. Operational impact  — containment, ETA, resource constraints

export function CrossModuleAlarms({ ckVals, fgVals, dirVals, rsVals, tcVals, tmVals, rhVals }) {
  const scenarios = [
    // ── Data quality (read first — contextualizes all downstream signals)
    scenarioDataSparseRisk(rsVals, tcVals),

    // ── Ignition context
    scenarioIgnitionTiming(fgVals, tcVals),
    scenarioOffHoursResponseRisk(fgVals, tcVals),

    // ── Fire behavior
    scenarioRelativeGrowthIntensity(fgVals),
    scenarioSurgeRisk(ckVals, fgVals, rsVals, tcVals),
    scenarioSpreadPressure(ckVals, fgVals, dirVals),
    scenarioFrontFragmentation(rsVals, fgVals),

    // ── Directional threat
    scenarioApproachConsistency(rsVals, tcVals),
    scenarioTrajectoryConfidence(rsVals, tcVals),
    scenarioFlankingThreat(ckVals, fgVals),

    // ── Operational impact
    scenarioProjectedAdvance(rsVals),
    scenarioETA(ckVals, { ...dirVals, ...rsVals }),
    scenarioContainmentDifficulty(fgVals, rsVals, rhVals),
  ];

  return (
    <div style={{ marginBottom: 14 }}>
      {scenarios.map((sc) => (
        <CrossModuleAlarmCard key={sc.title} scenario={sc} />
      ))}
    </div>
  );
}