#models/utils/enrich_features.py

import pandas as pd
import numpy as np

def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    EPS = 0.01  # Small epsilon to avoid division by zero

    # -------------------------------------------------------------------------
    # ETA Estimates
    # Estimated time of arrival based on various speed/growth proxies (log-compressed)
    # -------------------------------------------------------------------------
    df["eta_closing"]     = np.log1p(df["dist_min_ci_0_5h"] / df["closing_speed_m_per_h"].clip(lower=EPS))
    df["eta_radial"]      = np.log1p(df["dist_min_ci_0_5h"] / df["radial_growth_rate_m_per_h"].clip(lower=EPS))
    df["eta_projected"]   = np.log1p(df["dist_min_ci_0_5h"] / df["projected_advance_m"].clip(lower=EPS))
    df["eta_along_track"] = np.log1p(df["dist_min_ci_0_5h"] / df["along_track_speed"].clip(lower=EPS))

    # Combined ETA: total advance rate = closing speed + radial growth
    combined_speed = (df["closing_speed_m_per_h"].clip(lower=0) +
                      df["radial_growth_rate_m_per_h"].clip(lower=0)).clip(lower=EPS)
    df["eta_combined"] = np.log1p(df["dist_min_ci_0_5h"] / combined_speed)

    # -------------------------------------------------------------------------
    # Ratio Features
    # -------------------------------------------------------------------------
    df["dist_per_area"]   = np.log1p(df["dist_min_ci_0_5h"] / df["area_first_ha"].clip(lower=EPS))
    df["growth_pressure"] = np.log1p(df["area_growth_rate_ha_per_h"] / df["dist_min_ci_0_5h"].clip(lower=EPS))

    # -------------------------------------------------------------------------
    # Log-Transformed Distance & Speed Features
    # log1p stabilises skewed distributions and compresses large outliers
    # -------------------------------------------------------------------------
    df["log1p_dist_min"]           = np.log1p(df["dist_min_ci_0_5h"])
    df["log1p_dist_std"]           = np.log1p(df["dist_std_ci_0_5h"])
    df["log1p_dist_change"]        = np.log1p(np.abs(df["dist_change_ci_0_5h"]))
    df["log1p_projected"]          = np.log1p(df["projected_advance_m"].clip(lower=0))
    df["log1p_closing_speed"]      = np.log1p(df["closing_speed_abs_m_per_h"])
    df["log1p_centroid_speed"]     = np.log1p(df["centroid_speed_m_per_h"])
    df["log1p_radial_growth"]      = np.log1p(df["radial_growth_m"].clip(lower=0))
    df["log1p_radial_growth_rate"] = np.log1p(df["radial_growth_rate_m_per_h"].clip(lower=0))
    df["log1p_centroid_disp"]      = np.log1p(df["centroid_displacement_m"].clip(lower=0))
    df["log1p_dist_slope"]         = np.log1p(np.abs(df["dist_slope_ci_0_5h"]))

    # -------------------------------------------------------------------------
    # Directional Speed Split
    # Decompose closing speed into approach (positive) and recession (negative) components
    # -------------------------------------------------------------------------
    df["closing_pos"]    = df["closing_speed_m_per_h"].clip(lower=0)
    df["closing_neg"]    = df["closing_speed_m_per_h"].clip(upper=0).abs()
    df["is_approaching"] = (df["closing_speed_m_per_h"] > 0).astype(int)

    # -------------------------------------------------------------------------
    # Slope Quality Indicators
    # -------------------------------------------------------------------------
    # Weight slope by R² fit quality; high R² → reliable trend signal
    df["reliable_slope"]       = df["dist_slope_ci_0_5h"] * df["dist_fit_r2_0_5h"]
    df["log1p_reliable_slope"] = np.log1p(np.abs(df["reliable_slope"]))

    # Coefficient of variation: relative distance uncertainty
    df["dist_cv"] = df["dist_std_ci_0_5h"] / df["dist_min_ci_0_5h"].clip(lower=1)

    # Normalised acceleration: acceleration relative to current distance
    df["relative_accel"] = df["dist_accel_m_per_h2"] / df["dist_min_ci_0_5h"].clip(lower=1)

    # -------------------------------------------------------------------------
    # Linear-Fit ETA
    # Only valid when slope is sufficiently negative (fire is closing in)
    # -------------------------------------------------------------------------
    SLOPE_THRESHOLD = -0.1
    MAX_ETA_HOURS   = 500

    raw_eta = (-df["dist_min_ci_0_5h"] / df["dist_slope_ci_0_5h"].clip(upper=SLOPE_THRESHOLD)).clip(
        lower=0, upper=MAX_ETA_HOURS
    )
    df["linear_eta_ci"]  = np.where(df["dist_slope_ci_0_5h"] < SLOPE_THRESHOLD, np.log1p(raw_eta), np.nan)

    # Scale linear ETA by R² — penalises ETAs derived from poor linear fits
    df["reliable_eta_ci"] = df["linear_eta_ci"] * df["dist_fit_r2_0_5h"]

    # -------------------------------------------------------------------------
    # Alignment-Based Threat Features
    # corr(eta_aligned_along, event) ≈ 0.72  |  corr(eta_bearing_adjusted) ≈ 0.67
    # -------------------------------------------------------------------------

    # Along-track speed weighted by how well the fire bearing aligns with the CI direction.
    # When alignment_abs ≈ 1 the fire is heading almost straight at the CI;
    # when ≈ 0 the fire is moving perpendicular, so along_track_speed overstates the threat.
    df["aligned_along_speed"] = df["along_track_speed"] * df["alignment_abs"]
    df["eta_aligned_along"] = np.log1p(
        df["dist_min_ci_0_5h"] / df["aligned_along_speed"].clip(lower=EPS)
    )

    # Bearing-adjusted radial ETA:
    # spread_bearing_deg = 0 means the fire is spreading directly toward the CI.
    # cos(bearing) projects radial growth onto the approach axis.
    bearing_rad = np.deg2rad(df["spread_bearing_deg"].abs())
    df["bearing_threat_factor"]  = np.cos(bearing_rad).clip(lower=0)
    df["radial_bearing_threat"]  = (
        df["radial_growth_rate_m_per_h"].clip(lower=0) * df["bearing_threat_factor"]
    )
    df["eta_bearing_adjusted"] = np.log1p(
        df["dist_min_ci_0_5h"] / df["radial_bearing_threat"].clip(lower=EPS)
    )

    # -------------------------------------------------------------------------
    # Directional Decomposition
    # -------------------------------------------------------------------------

    # Fraction of total speed that is lateral (cross-track) vs. closing.
    # Low ratio → mostly head-on; high → flanking spread.
    df["cross_track_ratio"] = (
        np.abs(df["cross_track_component"]) / df["closing_speed_abs_m_per_h"].clip(lower=EPS)
    )

    # Direct interaction: alignment quality × positive closing speed.
    # Captures events where the fire moves toward the CI *and* its spread bearing matches.
    df["threat_alignment"]       = df["alignment_abs"] * df["closing_speed_m_per_h"].clip(lower=0)
    df["log1p_threat_alignment"] = np.log1p(df["threat_alignment"])

    # -------------------------------------------------------------------------
    # Compound Growth-Distance Threat
    # corr(area_radial_threat, event) ≈ 0.23
    # -------------------------------------------------------------------------

    # Joint effect of areal and radial growth relative to current distance.
    # A fire that is both expanding in area AND growing radially near the CI is high-risk.
    df["area_radial_threat"] = np.log1p(
        (df["area_growth_rate_ha_per_h"] * df["radial_growth_rate_m_per_h"].clip(lower=0)) /
        df["dist_min_ci_0_5h"].clip(lower=EPS)
    )

    # -------------------------------------------------------------------------
    # Reliability-Gated Acceleration
    # -------------------------------------------------------------------------

    # Weight acceleration by R²: noisy fits can produce spurious acceleration estimates.
    df["reliable_accel"] = df["dist_accel_m_per_h2"] * df["dist_fit_r2_0_5h"]

    # -------------------------------------------------------------------------
    # Speed Consistency
    # corr(log1p_speed_consistency, event) ≈ 0.24
    # -------------------------------------------------------------------------

    # Ratio of fire body movement (centroid speed) to CI-relative closing speed.
    # Values near 1 → centroid and CI-relative motion agree; outliers suggest asymmetric spread.
    df["speed_consistency"]       = df["centroid_speed_m_per_h"] / df["closing_speed_abs_m_per_h"].clip(lower=EPS)
    df["log1p_speed_consistency"] = np.log1p(df["speed_consistency"].clip(upper=100))

    # -------------------------------------------------------------------------
    # Temporal / Diurnal Risk
    # -------------------------------------------------------------------------

    # Fires starting during peak diurnal wind hours (10:00–18:00 local) tend to
    # spread faster due to low relative humidity and stronger winds.
    hour = df["event_start_hour"]
    df["is_high_wind_hour"] = ((hour >= 10) & (hour <= 18)).astype(int)

    # Cyclical encoding: avoids ordinal distance artefact at midnight boundary.
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # -------------------------------------------------------------------------
    # Normalised Distance Change (momentum proxy)
    # -------------------------------------------------------------------------

    # Fractional change in distance over the observation window.
    # Negative value → distance is shrinking (fire closing in).
    df["dist_change_norm"] = df["dist_change_ci_0_5h"] / df["dist_min_ci_0_5h"].clip(lower=1)

    # -------------------------------------------------------------------------
    # Area–Speed Ratio
    # corr(area_speed_ratio, event) ≈ 0.32
    # -------------------------------------------------------------------------

    # Large, slow-moving fires can still threaten structures if they are close;
    # this ratio captures that relationship.
    df["area_speed_ratio"] = np.log1p(
        df["area_first_ha"] / df["centroid_speed_m_per_h"].clip(lower=EPS)
    )

    # "Alignment × observation depth": the fire is heading directly toward the CI
    # AND has been tracked multiple times → the threat is more confirmed
    df["alignment_x_nperim"] = df["alignment_abs"] * df["num_perimeters_0_5h"]

    # "Temporal momentum × alignment": the fire has been active for a longer duration
    # AND is heading toward the CI → less likely to change direction
    df["dt_x_alignment"] = df["dt_first_last_0_5h"] * df["alignment_abs"]

    # "Directional momentum": the fire body is moving quickly
    # AND in the direction of the CI
    df["momentum_threat"] = df["alignment_abs"] * df["centroid_speed_m_per_h"]

    # "Growth × alignment": the fire is expanding rapidly
    # AND toward the CI
    df["directional_growth"] = df["alignment_abs"] * df["area_growth_rate_ha_per_h"]

    return df