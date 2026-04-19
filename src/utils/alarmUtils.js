// src/utils/alarmUtils.js
// Utility functions for alarm level calculation and formatting

export function clamp01(v) { return Math.min(1, Math.max(0, v)); }

export function alarmLevel(score) {
  if (score >= 0.5) return "alarm";
  if (score >= 0.2) return "watch";
  return "ok";
}

export const BADGE_STYLES = {
  ok:    { label: "OK",    bg: "#EAF3DE", color: "#3B6D11" },
  watch: { label: "WATCH", bg: "#FAEEDA", color: "#854F0B" },
  alarm: { label: "ALARM", bg: "#FCEBEB", color: "#A32D2D" },
};

export const ALARM_DOT_COLORS = {
  ok:    "#639922",
  watch: "#BA7517",
  alarm: "#A32D2D",
};

export function formatTick(v) {
  if (Math.abs(v) < 1e-6) return "0";
  if (Math.abs(v - 1) < 1e-6) return "1";
  return parseFloat(v.toFixed(2)).toString();
}

export function fmtVal(field, v, FEATURE_RANGES) {
  const unit = field.unit || "";
  if (["m", "m/h", "m/h²", "ha/h", "ha"].includes(unit)) {
    const sign = v > 0 && FEATURE_RANGES[field.id]?.min < 0 ? "+" : "";
    const abs = Math.abs(v);
    return (v < 0 ? "−" : sign) + (abs >= 1000 ? (abs / 1000).toFixed(1) + "k" : Math.round(abs)) + (unit ? " " + unit : "");
  }
  if (unit === "°") return Math.round(v) + "°";
  if (unit === "%") return Math.round(v) + "%";
  return parseFloat(v).toFixed(2) + (unit ? " " + unit : "");
}

export function fmtFieldVal(field, v, FEATURE_RANGES) {
  const rng = FEATURE_RANGES[field.id];
  if (!rng) return String(v);
  return rng.step < 1 ? parseFloat(v).toFixed(rng.step < 0.01 ? 3 : 2) : String(Math.round(v));
}

export function applyFGDerived(vals, id, v) {
  const next = { ...vals, [id]: v };
  if (id === "area_growth_rel_0_5h") next["relative_growth_0_5h"] = v;
  if (id === "area_first_ha") next["log1p_area_first"] = parseFloat(Math.log1p(v).toFixed(2));
  if (id === "area_growth_abs_0_5h") {
    next["log1p_growth"] = parseFloat(Math.log1p(v).toFixed(2));
    next["area_growth_rate_ha_per_h"] = parseFloat((v / 5).toFixed(1));
    const a0 = next["area_first_ha"];
    if (a0 > 0) {
      const af = a0 + v;
      next["radial_growth_m"] = Math.round((Math.sqrt(af / Math.PI) - Math.sqrt(a0 / Math.PI)) * 100);
      next["radial_growth_rate_m_per_h"] = Math.round(next["radial_growth_m"] / 5);
      next["area_growth_rel_0_5h"] = parseFloat((v / a0).toFixed(2));
      next["relative_growth_0_5h"] = next["area_growth_rel_0_5h"];
      next["log_area_ratio_0_5h"] = parseFloat(Math.log(af / a0).toFixed(3));
    }
  }
  return next;
}