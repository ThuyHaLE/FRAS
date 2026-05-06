// dashboard/src/data/mockData.js
// This file contains mock data for testing and development purposes. 
// The data structure mimics the expected output from the API, 
// allowing developers to work with realistic data without needing to make actual API calls.

const MOCK_DATA = {
  event_id: 49565135,
  centroidKinematics: {
    centroid_displacement_m: 208.2907737689164,
    centroid_speed_m_per_h: 44.85321137084563,
    spread_bearing_deg: 238.48783085358303,
    spread_bearing_sin: -0.8525291707796858,
    spread_bearing_cos: -0.522679646599809,
  },
  growthFeatures: {
    area_first_ha: 71.05631404185833,
    area_growth_abs_0_5h: 38.43712325113634,
    area_growth_rel_0_5h: 0.5409388844528797,
    area_growth_rate_ha_per_h: 8.277027265658672,
    log1p_area_first: 4.27744795277463,
    log1p_growth: 3.6747075871942823,
    log_area_ratio_0_5h: 0.4323918909439588,
    relative_growth_0_5h: 0.5409388844528797,
    radial_growth_m: 114.77988269138484,
    radial_growth_rate_m_per_h: 24.71663168906923,
  },
  directionality: {
    alignment_cos: -0.0540944819115652,
    alignment_abs: 0.0540944819115652,
    cross_track_component: 44.787538267772945,
    along_track_speed: -2.4263112311758204,
  },
  distanceToEvacuationZoneCentroids: {
    dist_min_ci_0_5h: 839.1486030725575,
    dist_std_ci_0_5h: 1.1368683772161605e-13,
    dist_change_ci_0_5h: 0.0,
    dist_slope_ci_0_5h: 0.0,
    closing_speed_m_per_h: 0.0,
    closing_speed_abs_m_per_h: 0.0,
    projected_advance_m: 0.0,
    dist_accel_m_per_h2: -4.3980432255329913e-13,
    dist_fit_r2_0_5h: 0.0,
  },
  reachProbability: {
    prob_12h: 0.99, prob_24h: 1.0, prob_48h: 1.0, prob_72h: 1.0,
  },
  temporalCoverage: {
    num_perimeters_0_5h: 11,
    dt_first_last_0_5h: 4.643831899722222,
    low_temporal_resolution_0_5h: 0,
  },
  temporalMetadata: {
    event_start_hour: 22,
    event_start_dayofweek: 3,
    event_start_month: 1,
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