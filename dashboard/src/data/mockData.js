// dashboard/src/data/mockData.js
// This file contains mock data for testing and development purposes. 
// The data structure mimics the expected output from the API, 
// allowing developers to work with realistic data without needing to make actual API calls.

const MOCK_DATA = {
  event_id: 35311039,
  centroidKinematics: {
    centroid_displacement_m: 0.274795128413165,
    centroid_speed_m_per_h: 0.0733292203160629,
    spread_bearing_deg: 261.95231624470296,
    spread_bearing_sin: -0.9901519005978356,
    spread_bearing_cos: -0.1399971919093157,
  },
  growthFeatures: {
    area_first_ha: 403.6517949491382,
    log1p_area_first: 6.003026931738467,
    area_growth_abs_0_5h: 0.3754430478697941,
    log1p_growth: 0.3187758958504815,
    area_growth_rel_0_5h: 0.0009301161361541,
    relative_growth_0_5h: 0.0009301161361541,
    log_area_ratio_0_5h: 0.0009296838438712,
    area_growth_rate_ha_per_h: 0.1001871690097229,
    radial_growth_m: 0.5270292752720707,
    radial_growth_rate_m_per_h: 0.1406380311856691,
  },
  directionality: {
    alignment_cos: 0.4537314696716237, alignment_abs: 0.4537314696716237,
    cross_track_component: 0.0653464884053239, along_track_speed: 0.0332717749038815,
  },
  distanceToEvacuationZoneCentroids: {
    dist_min_ci_0_5h: 1204.296339766883,
    dist_std_ci_0_5h: 0.0, dist_change_ci_0_5h: 0.0,
    dist_slope_ci_0_5h: -1.7029183079261637e-13,
    closing_speed_m_per_h: 0.0, dist_accel_m_per_h2: -3.209511122563853e-13,
    projected_advance_m: 0.0, closing_speed_abs_m_per_h: 0.0, dist_fit_r2_0_5h: 0.0,
  },
  reachProbability: {
    prob_12h: 0.666636, prob_24h: 0.860151, prob_48h: 0.860151, prob_72h: 0.939452,
  },
  temporalCoverage: {
    num_perimeters_0_5h: 12, dt_first_last_0_5h: 3.7474164763888886, low_temporal_resolution_0_5h: 0.0,
  },
  temporalMetadata: {
    event_start_hour: 21, event_start_dayofweek: 4, event_start_month: 8,
  },
};

export async function fetchAllModules() {
  await new Promise((r) => setTimeout(r, 0));
  const d = MOCK_DATA;
  return {
    eventId:            d.event_id,
    centroidKinematics: { ...d.centroidKinematics },
    fireGrowth:         { ...d.growthFeatures },
    directionality:     { ...d.directionality },
    riskScore:          { ...d.distanceToEvacuationZoneCentroids },
    reachProbability:   { ...d.reachProbability },
    temporalCoverage:   { ...d.temporalCoverage },
    temporalMetadata:   { ...d.temporalMetadata },
  };
}