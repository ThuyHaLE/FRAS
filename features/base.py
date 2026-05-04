# features/base.py
HORIZONS       = [12, 24, 48, 72]

BRIER_W        = {24: 0.3, 48: 0.4, 72: 0.3}

CINDEX_HORIZON = 48

RANDOM_STATE   = 42

BASE_FEATURES = [
    "num_perimeters_0_5h",          # Number of perimeters within first 5 hours
    "dt_first_last_0_5h",           # Time span between first and last perimeter (hours)
    "low_temporal_resolution_0_5h", # Flag, 1 if dt < 0.5h or only 1 perimeter, else 0
    "area_first_ha",                # Initial fire area at t0 (hectares)
    "area_growth_abs_0_5h",         # Absolute area growth (hectares)
    "area_growth_rel_0_5h",         # Relative area growth (fraction)
    "area_growth_rate_ha_per_h",    # Area growth rate (hectares/hour)
    "log1p_area_first",             # Log(1 + initial area)
    "log1p_growth",                 # Log(1 + absolute growth)
    "log_area_ratio_0_5h",          # Log ratio of final to initial area
    "relative_growth_0_5h",         # Relative growth (same as area_growth_rel_0_5h)
    "radial_growth_m",              # Change in effective radius (meters)
    "radial_growth_rate_m_per_h",   # Rate of radial growth (meters/hour)
    "centroid_displacement_m",      # Total displacement of fire centroid (meters)
    "centroid_speed_m_per_h",       # Speed of centroid movement (meters/hour)
    "spread_bearing_deg",           # Bearing/direction of fire spread (degrees)
    "spread_bearing_sin",           # Sine of spread bearing (circular encoding)
    "spread_bearing_cos",           # Cosine of spread bearing (circular encoding)
    "dist_min_ci_0_5h",             # Minimum distance to nearest evac zone centroid (meters)
    "dist_std_ci_0_5h",             # Standard deviation of distances
    "dist_change_ci_0_5h",          # Change in distance (d5 - d0, negative = closing)
    "dist_slope_ci_0_5h",           # Linear slope of distance vs time (meters/hour)
    "closing_speed_m_per_h",        # Speed at which fire is closing distance (m/hour, positive = closing)
    "closing_speed_abs_m_per_h",    # Absolute closing speed
    "projected_advance_m",          # Projected advance toward evac zone (d0 - d5)
    "dist_accel_m_per_h2",          # Acceleration in distance change (meters/hour^2)
    "dist_fit_r2_0_5h",             # R^2 of linear fit to distance vs time
    "alignment_cos",                # Cosine of angle between fire motion and evac direction
    "alignment_abs",                # Absolute alignment between fire motion and evac direction (0-1, higher = more aligned)
    "cross_track_component",        # Sideways drift component
    "along_track_speed",            # Speed component toward/away from evac
    "event_start_hour",             # Hour of day when fire started (0-23)
    "event_start_dayofweek",        # Day of week (0 = Monday, 6 = Sunday)
    "event_start_month"             # Month when fire started (1-12)
    ]