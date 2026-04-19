// src/components/editor/ParamEditor.jsx
// A component for adjusting feature parameters via sliders and number inputs, 
// with support for uploading JSON files to set multiple parameters at once.

import { useState, useEffect } from "react";
import { FEATURE_RANGES, ACCENT } from "./constants/featureRanges";
import { fmtFieldVal, applyFGDerived } from "./utils/alarmUtils";
import { JsonUploadModal } from "./JsonUploadModal";

const EDITOR_MODULES = [
  { key: "ck", label: "Centroid Kinematics", dotColor: ACCENT,
    fields: [
      { id: "centroid_displacement_m", label: "centroid_displacement_m", unit: "m",   readonly: false },
      { id: "centroid_speed_m_per_h",  label: "centroid_speed_m_per_h",  unit: "m/h", readonly: false },
      { id: "spread_bearing_deg",      label: "spread_bearing_deg",      unit: "°",   readonly: false },
      { id: "spread_bearing_sin",      label: "spread_bearing_sin",      unit: "",    readonly: true  },
      { id: "spread_bearing_cos",      label: "spread_bearing_cos",      unit: "",    readonly: true  },
    ],
  },
  { key: "fg", label: "Growth Features", dotColor: "#D85A30",
    fields: [
      { id: "area_first_ha",              label: "area_first_ha",              unit: "ha",   readonly: false },
      { id: "log1p_area_first",           label: "log1p_area_first",           unit: "",     readonly: true  },
      { id: "area_growth_abs_0_5h",       label: "area_growth_abs_0_5h",       unit: "ha",   readonly: false },
      { id: "log1p_growth",               label: "log1p_growth",               unit: "",     readonly: true  },
      { id: "area_growth_rel_0_5h",       label: "area_growth_rel_0_5h",       unit: "×",    readonly: false },
      { id: "relative_growth_0_5h",       label: "relative_growth_0_5h",       unit: "×",    readonly: true  },
      { id: "log_area_ratio_0_5h",        label: "log_area_ratio_0_5h",        unit: "",     readonly: true  },
      { id: "area_growth_rate_ha_per_h",  label: "area_growth_rate_ha_per_h",  unit: "ha/h", readonly: false },
      { id: "radial_growth_m",            label: "radial_growth_m",            unit: "m",    readonly: false },
      { id: "radial_growth_rate_m_per_h", label: "radial_growth_rate_m_per_h", unit: "m/h",  readonly: false },
    ],
  },
  { key: "dir", label: "Directionality", dotColor: "#7F77DD",
    fields: [
      { id: "alignment_cos",         label: "alignment_cos",     unit: "",    readonly: false },
      { id: "alignment_abs",         label: "alignment_abs",     unit: "",    readonly: false },
      { id: "cross_track_component", label: "cross_track",       unit: "m/h", readonly: false },
      { id: "along_track_speed",     label: "along_track_speed", unit: "m/h", readonly: false },
    ],
  },
  { key: "rs", label: "Distance to Evacuation Zone Centroids", dotColor: "#378ADD",
    fields: [
      { id: "dist_min_ci_0_5h",          label: "dist_min_ci_0_5h",          unit: "m",    readonly: false },
      { id: "dist_std_ci_0_5h",          label: "dist_std_ci_0_5h",          unit: "m",    readonly: false },
      { id: "dist_change_ci_0_5h",       label: "dist_change_ci_0_5h",       unit: "m",    readonly: false },
      { id: "dist_slope_ci_0_5h",        label: "dist_slope_ci_0_5h",        unit: "m/h",  readonly: false },
      { id: "closing_speed_m_per_h",     label: "closing_speed_m_per_h",     unit: "m/h",  readonly: false },
      { id: "dist_accel_m_per_h2",       label: "dist_accel_m_per_h2",       unit: "m/h²", readonly: false },
      { id: "projected_advance_m",       label: "projected_advance_m",       unit: "m",    readonly: false },
      { id: "closing_speed_abs_m_per_h", label: "closing_speed_abs_m_per_h", unit: "m/h",  readonly: false },
      { id: "dist_fit_r2_0_5h",          label: "dist_fit_r2_0_5h",          unit: "",     readonly: false },
    ],
  },
  { key: "tc", label: "Temporal Coverage", dotColor: "#7F77DD",
    fields: [
      { id: "num_perimeters_0_5h",          label: "num_perimeters_0_5h",          unit: "",  readonly: false },
      { id: "dt_first_last_0_5h",           label: "dt_first_last_0_5h",           unit: "h", readonly: false },
      { id: "low_temporal_resolution_0_5h", label: "low_temporal_resolution_0_5h", unit: "",  readonly: true  },
    ],
  },
  { key: "tm", label: "Temporal Metadata", dotColor: "#D85A30",
    fields: [
      { id: "event_start_hour",      label: "event_start_hour",      unit: "h", readonly: false },
      { id: "event_start_dayofweek", label: "event_start_dayofweek", unit: "",  readonly: false },
      { id: "event_start_month",     label: "event_start_month",     unit: "",  readonly: false },
    ],
  },
];

const MODULE_KEY_MAP = {
  ck:  ["centroid_displacement_m","centroid_speed_m_per_h","spread_bearing_deg","spread_bearing_sin","spread_bearing_cos"],
  fg:  ["area_first_ha","log1p_area_first","area_growth_abs_0_5h","log1p_growth","area_growth_rel_0_5h","relative_growth_0_5h","log_area_ratio_0_5h","area_growth_rate_ha_per_h","radial_growth_m","radial_growth_rate_m_per_h"],
  dir: ["alignment_cos","alignment_abs","cross_track_component","along_track_speed"],
  rs:  ["dist_min_ci_0_5h","dist_std_ci_0_5h","dist_change_ci_0_5h","dist_slope_ci_0_5h","closing_speed_m_per_h","dist_accel_m_per_h2","projected_advance_m","closing_speed_abs_m_per_h","dist_fit_r2_0_5h"],
  tc:  ["num_perimeters_0_5h","dt_first_last_0_5h","low_temporal_resolution_0_5h"],
  tm:  ["event_start_hour","event_start_dayofweek","event_start_month"],
};

const NESTED_MAP = {
  centroidKinematics: "ck", fireGrowth: "fg", directionality: "dir",
  riskScore: "rs", temporalCoverage: "tc", temporalMetadata: "tm",
};

function applyJsonToEditorMap(parsed, editorMap) {
  const flat = {};
  Object.entries(NESTED_MAP).forEach(([jsonKey]) => {
    if (parsed[jsonKey] && typeof parsed[jsonKey] === "object") Object.assign(flat, parsed[jsonKey]);
  });
  Object.entries(parsed).forEach(([k, v]) => { if (typeof v === "number") flat[k] = v; });
  const updates = {};
  Object.entries(MODULE_KEY_MAP).forEach(([modKey, fields]) => {
    const [vals] = editorMap[modKey];
    const patch = {};
    fields.forEach((fid) => { if (flat[fid] !== undefined) patch[fid] = flat[fid]; });
    if (Object.keys(patch).length) updates[modKey] = { ...vals, ...patch };
  });
  if (updates.fg) {
    let fg = updates.fg;
    if (fg.area_first_ha !== undefined)        fg = applyFGDerived(fg, "area_first_ha", fg.area_first_ha);
    if (fg.area_growth_abs_0_5h !== undefined) fg = applyFGDerived(fg, "area_growth_abs_0_5h", fg.area_growth_abs_0_5h);
    updates.fg = fg;
  }
  if (updates.tc) updates.tc.low_temporal_resolution_0_5h = (updates.tc.dt_first_last_0_5h < 0.5 || updates.tc.num_perimeters_0_5h <= 1) ? 1 : 0;
  return updates;
}

function EditorField({ field, value, onChange }) {
  const rng = FEATURE_RANGES[field.id] ?? { min: 0, max: 1, step: 0.01 };
  const [inputVal, setInputVal] = useState(fmtFieldVal(field, value, FEATURE_RANGES));
  useEffect(() => { setInputVal(fmtFieldVal(field, value, FEATURE_RANGES)); }, [value]);
  const clamp = (v) => Math.min(rng.max, Math.max(rng.min, v));
  const handleInput = (e) => { setInputVal(e.target.value); const n = parseFloat(e.target.value); if (!isNaN(n)) onChange(clamp(n)); };
  const handleBlur  = () => { const n = parseFloat(inputVal); const c = isNaN(n) ? value : clamp(n); onChange(c); setInputVal(fmtFieldVal(field, c, FEATURE_RANGES)); };
  const ro = field.readonly;
  return (
    <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1.6fr) minmax(0,2fr) 64px", alignItems: "center", gap: 8, padding: "5px 0", borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
      <div>
        <div style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: ro ? "var(--color-text-tertiary)" : "var(--color-text-secondary)", lineHeight: 1.3 }}>{field.label}</div>
        {field.unit && <div style={{ fontSize: 9, color: "var(--color-text-tertiary)" }}>{field.unit}</div>}
      </div>
      {ro ? <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", fontStyle: "italic" }}>auto-computed</div>
           : <input type="range" min={rng.min} max={rng.max} step={rng.step} value={value} onChange={(e) => onChange(clamp(parseFloat(e.target.value)))} style={{ width: "100%", accentColor: ACCENT }} />}
      <input type="number" min={rng.min} max={rng.max} step={rng.step} value={inputVal} disabled={ro} onChange={handleInput} onBlur={handleBlur}
        style={{ width: "100%", fontSize: 11, fontFamily: "var(--font-mono)", padding: "3px 6px", borderRadius: 4, boxSizing: "border-box", border: "0.5px solid var(--color-border-secondary)", background: ro ? "var(--color-background-secondary)" : "var(--color-background-primary)", color: ro ? "var(--color-text-tertiary)" : "var(--color-text-primary)", textAlign: "right" }}
      />
    </div>
  );
}

function EditorModuleSection({ mod, vals, onChange }) {
  const [open, setOpen] = useState(false);
  function handleChange(fieldId, newVal) {
    const next = { ...vals, [fieldId]: newVal };
    if (mod.key === "ck" && fieldId === "spread_bearing_deg") {
      next.spread_bearing_sin = parseFloat(Math.sin((newVal * Math.PI) / 180).toFixed(3));
      next.spread_bearing_cos = parseFloat(Math.cos((newVal * Math.PI) / 180).toFixed(3));
    }
    if (mod.key === "fg") return onChange(applyFGDerived(vals, fieldId, newVal));
    if (mod.key === "tc") next.low_temporal_resolution_0_5h = (next.dt_first_last_0_5h < 0.5 || next.num_perimeters_0_5h <= 1) ? 1 : 0;
    onChange(next);
  }
  return (
    <div style={{ borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
      <button onClick={() => setOpen((p) => !p)} style={{ width: "100%", display: "flex", alignItems: "center", gap: 8, padding: "8px 12px", background: "transparent", border: "none", cursor: "pointer", textAlign: "left" }}>
        <div style={{ width: 8, height: 8, borderRadius: 2, background: mod.dotColor, flexShrink: 0 }} />
        <span style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-primary)", flex: 1 }}>{mod.label}</span>
        <span style={{ fontSize: 11, color: "var(--color-text-tertiary)" }}>{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <div style={{ padding: "4px 12px 10px" }}>
          {mod.fields.map((f) => <EditorField key={f.id} field={f} value={vals?.[f.id] ?? 0} onChange={(v) => handleChange(f.id, v)} />)}
        </div>
      )}
    </div>
  );
}

export function ParamEditor({ editorMap }) {
  const [open, setOpen]           = useState(false);
  const [showUpload, setShowUpload] = useState(false);
  function handleApplyJson(parsed) {
    const updates = applyJsonToEditorMap(parsed, editorMap);
    Object.entries(updates).forEach(([modKey, newVals]) => { const [, setVals] = editorMap[modKey]; setVals(newVals); });
  }
  return (
    <>
      {showUpload && <JsonUploadModal onApply={handleApplyJson} onClose={() => setShowUpload(false)} />}
      <div style={{ background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: 10, marginBottom: 14, overflow: "hidden" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "10px 16px" }}>
          <span style={{ fontSize: 13 }}>⚙</span>
          <span style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-primary)", flex: 1 }}>Adjust parameters</span>
          <button onClick={() => setShowUpload(true)} style={{ fontSize: 11, padding: "3px 10px", borderRadius: 5, border: "0.5px solid var(--color-border-secondary)", background: "transparent", color: "var(--color-text-secondary)", cursor: "pointer", marginRight: 6 }}>↑ Upload JSON</button>
          <span style={{ fontSize: 10, color: "var(--color-text-tertiary)", marginRight: 8 }}>Slider + số — thay đổi realtime</span>
          <button onClick={() => setOpen((p) => !p)} style={{ background: "none", border: "none", cursor: "pointer", fontSize: 11, color: "var(--color-text-tertiary)", padding: 0 }}>{open ? "▲" : "▼"}</button>
        </div>
        {open && (
          <div style={{ borderTop: "0.5px solid var(--color-border-tertiary)" }}>
            {EDITOR_MODULES.map((mod) => {
              const [vals, setVals] = editorMap[mod.key];
              return <EditorModuleSection key={mod.key} mod={mod} vals={vals} onChange={setVals} />;
            })}
          </div>
        )}
      </div>
    </>
  );
}