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
//   - Speed fast path: speed > 0.75 can trigger alarm independently of displacement
//    (early warning when fire starts moving fast, even if displacement is still small)
//   - Weighted sum: displacement×0.30 + speed×0.50 + interaction×0.20
//   - Math.max guard defensive pipeline sending negative values
export function useCkAlarm(vals) {
  const displacement = clamp01(
    Math.max(vals.centroid_displacement_m, 0) / R("centroid_displacement_m").max
  );
  const speed = clamp01(
    Math.max(vals.centroid_speed_m_per_h, 0) / R("centroid_speed_m_per_h").max
  );

  // Fast path: only speed high is enough to trigger early warning
  // (fire starts moving fast → displacement still low but velocity indicates danger)
  const speedAlarm = speed > 0.75 ? clamp01(speed * 0.90) : 0;

  // Weighted sum + interaction
  const mobility = clamp01(
    0.30 * displacement + 0.50 * speed + 0.20 * displacement * speed
  );

  // Get max of two paths: fast path and weighted sum
  const score = clamp01(Math.max(speedAlarm, mobility));
  const level = alarmLevel(score);

  return {
    overallScore: score,
    signals: [
      {
        name: "displacement",
        value: displacement,
        hint: `${Math.round(vals.centroid_displacement_m)} m total centroid shift`,
      },
      {
        name: "speed",
        value: speed,
        hint: `${Math.round(vals.centroid_speed_m_per_h)} m/h centroid speed`,
      },
      {
        name: "mobility",
        value: score,
        hint: `max(speedFastPath, 0.30×disp + 0.50×speed + 0.20×interaction)`,
      },
    ],
    summary: {
      ok:    "Fire centroid is stationary — no directional threat from movement yet.",
      watch: "Centroid is moving — monitor spread direction relative to evac zones.",
      alarm: "Rapid centroid movement detected — fire is spreading across terrain.",
    }[level],
  };
}

// ── Fire Growth
//   - Math.max(0, ...) before sqrt to avoid NaN if pipeline sends negative values
export function useFgAlarm(vals) {
  const size = clamp01(vals.log1p_area_first / R("log1p_area_first").max);

  // Guard: sqrt of negative values → NaN; clamp to 0 before
  const rate = clamp01(
    Math.sqrt(Math.max(vals.area_growth_rate_ha_per_h, 0) / R("area_growth_rate_ha_per_h").max)
  );

  const rawRel = Math.max(vals.log_area_ratio_0_5h, 0);
  const rel    = clamp01(rawRel / R("log_area_ratio_0_5h").max);

  const intensity = clamp01(0.25 * size + 0.45 * rate + 0.30 * rate * rel);
  const level     = alarmLevel(intensity);

  return {
    overallScore: intensity,
    signals: [
      {
        name: "initial size",
        value: size,
        hint: `${vals.area_first_ha.toFixed(0)} ha baseline (log-scaled)`,
      },
      {
        name: "growth rate",
        value: rate,
        hint: `${vals.area_growth_rate_ha_per_h.toFixed(1)} ha/h (sqrt-scaled)`,
      },
      {
        name: "rel. burst",
        value: rel,
        hint: `log_area_ratio = ${vals.log_area_ratio_0_5h.toFixed(2)} — ${
          rawRel > 0
            ? `fire is ${Math.exp(rawRel).toFixed(1)}× initial size`
            : "no net expansion (noise/correction)"
        }`,
      },
      {
        name: "intensity",
        value: intensity,
        hint: "0.25×size + 0.45×rate + 0.30×(rate × rel)",
      },
    ],
    summary: {
      ok:    "Fire size and growth rate are within manageable range.",
      watch: "Fire is large or expanding — growth signals warrant close monitoring.",
      alarm: "Rapid fire expansion detected — containment window is narrowing.",
    }[level],
  };
}

// ── Directionality
//    Gate directly on `vals.alignment_cos < 0.15`.
export function useDirAlarm(vals) {
  const directness = clamp01((vals.alignment_cos - 0.15) / 0.85);

  const crossCeil = Math.max(
    Math.abs(R("cross_track_component").min),
    R("cross_track_component").max
  );
  const focus = clamp01(
    1 - Math.pow(Math.abs(vals.cross_track_component) / crossCeil, 0.6)
  );

  const orientation = clamp01(directness * 0.60 + focus * 0.40);

  const approachRaw = Math.max(vals.along_track_speed, 0);
  const approach    = clamp01(approachRaw / R("along_track_speed").max);

  // Gate on raw alignment_cos — filter real movement tangential
  // (alignment_cos < 0.15 ≈ angle > 81° compare with evac zone direction)
  const alarm =
    vals.alignment_cos < 0.15
      ? 0
      : clamp01(orientation * 0.50 + approach * 0.50);

  const level = alarmLevel(alarm);

  return {
    overallScore: alarm,
    signals: [
      {
        name: "alignment",
        value: directness,
        hint: `alignment_cos = ${vals.alignment_cos.toFixed(2)} → ${
          vals.alignment_cos > 0.7
            ? "direct"
            : vals.alignment_cos > 0.3
            ? "oblique"
            : "neutral"
        }`,
      },
      {
        name: "focus (1−drift)",
        value: focus,
        hint: `cross_track = ${vals.cross_track_component.toFixed(0)} m/h`,
      },
      {
        name: "approach speed",
        value: approach,
        hint: `along_track = ${approachRaw.toFixed(0)} m/h toward zones`,
      },
      {
        name: "dir. alarm",
        value: alarm,
        hint:
          vals.alignment_cos < 0.15
            ? `gated out — alignment_cos ${vals.alignment_cos.toFixed(2)} < 0.15`
            : "orientation(align×focus)×0.5 + approach×0.5",
      },
    ],
    summary: {
      ok:    "Direction is neutral — fire is not oriented toward evacuation zones.",
      watch: "Fire shows partial alignment toward evac zones — track bearing changes.",
      alarm: "Fire is heading directly toward evacuation zones at speed.",
    }[level],
  };
}

// ── Risk Score (Proximity)
//     dist_accel negative = fire is increasing speed toward zone (most dangerous)
//     Flip sign: accelRisk = max(-dist_accel, 0)
//     Normalize vs abs(min) = 120 m/h²
//   - approachSignal rebalance: closing×conf×0.75 + accel×0.25
export function useRsAlarm(vals) {
  // ── Proximity composite
  const proximity = clamp01(
    1 - Math.pow(vals.dist_min_ci_0_5h / R("dist_min_ci_0_5h").max, 0.5)
  );

  const advanceRaw = Math.max(vals.projected_advance_m, 0);
  const advance    = clamp01(advanceRaw / R("projected_advance_m").max);

  const proxSignal = clamp01(proximity * 0.65 + advance * 0.35);

  // ── Approach composite
  const closingRaw = Math.max(vals.closing_speed_m_per_h, 0);
  const closing    = clamp01(closingRaw / R("closing_speed_m_per_h").max);

  const conf       = clamp01(vals.dist_fit_r2_0_5h);
  const confFactor = 0.35 + 0.65 * conf;

  // dist_accel negative = fire is increasing speed toward zone (most dangerous)
  // Flip sign to: the more negative the value → the higher the accelRisk
  // Normalize vs abs(min) = 120 m/h² (ceiling of acceleration toward the zone)
  const accelRisk = clamp01(
    Math.max(-vals.dist_accel_m_per_h2, 0) / Math.abs(R("dist_accel_m_per_h2").min)
  );

  // closing (confidence-weighted) as 75%, acceleration as 25%
  // Acceleration is a signal that comes earlier but is noisier → weight lower
  const approachSignal = clamp01(closing * confFactor * 0.75 + accelRisk * 0.25);

  // ── Final blend
  const alarm  = clamp01(proxSignal * 0.55 + approachSignal * 0.45);
  const distKm = (vals.dist_min_ci_0_5h / 1000).toFixed(1);
  const level  = alarmLevel(alarm);

  return {
    overallScore: alarm,
    signals: [
      {
        name: "proximity",
        value: proximity,
        hint: `${distKm} km to nearest zone (sqrt-inverted)`,
      },
      {
        name: "advance",
        value: advance,
        hint: `${advanceRaw.toFixed(0)} m projected advance`,
      },
      {
        name: "closing speed",
        value: closing,
        hint: `${closingRaw.toFixed(0)} m/h approach rate`,
      },
      {
        name: "acceleration",
        value: accelRisk,
        hint: `dist_accel = ${vals.dist_accel_m_per_h2.toFixed(1)} m/h² — ${
          vals.dist_accel_m_per_h2 < -20
            ? "fire accelerating toward zones"
            : vals.dist_accel_m_per_h2 < 0
            ? "slight closing acceleration"
            : "stable or decelerating"
        }`,
      },
      {
        name: "trajectory conf",
        value: conf,
        hint: `R² = ${vals.dist_fit_r2_0_5h.toFixed(2)} — ${
          conf > 0.7 ? "reliable" : conf > 0.3 ? "moderate" : "noisy"
        }`,
      },
      {
        name: "proximity alarm",
        value: alarm,
        hint: `proxSignal(${proxSignal.toFixed(2)})×0.55 + approachSignal(${approachSignal.toFixed(2)})×0.45`,
      },
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

    // ── Operational impact
    scenarioContainmentDifficulty(fgVals, rsVals, rhVals),
    scenarioProjectedAdvance(rsVals),
    scenarioETA(ckVals, { ...dirVals, ...rsVals }),
    
    // ── Ignition context
    scenarioIgnitionTiming(fgVals, tcVals),
    scenarioOffHoursResponseRisk(fgVals, tcVals, tmVals),

    // ── Fire behavior
    scenarioRelativeGrowthIntensity(fgVals),
    scenarioSurgeRisk(ckVals, fgVals, rsVals, tcVals),
    scenarioSpreadPressure(ckVals, fgVals, dirVals),
    scenarioFrontFragmentation(rsVals, fgVals),

    // ── Directional threat
    scenarioApproachConsistency(rsVals, tcVals),
    scenarioTrajectoryConfidence(rsVals, tcVals),
    scenarioFlankingThreat(ckVals, fgVals),
  ];

  return (
    <div style={{ marginBottom: 14 }}>
      {scenarios.map((sc) => (
        <CrossModuleAlarmCard key={sc.title} scenario={sc} />
      ))}
    </div>
  );
}