// dashboard/src/components/editor/ParamEditor.jsx
// A collapsible panel for adjusting feature parameters with sliders and numeric inputs, 
// and a Predict button to run the model with the current params. 
// This is meant for power users to understand how different features affect the model's predictions, 
// and to quickly test hypothetical scenarios by tweaking feature values.

import { useState, useEffect } from "react";
import { FEATURE_RANGES, ACCENT } from "../../constants/featureRanges";
import { fmtFieldVal, applyFGDerived } from "../../utils/alarmUtils";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

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

// Collect all feature values from the editorMap into a single event object to send to the API for prediction.
function collectEvent(editorMap) {
  const event = {
    event_id: Math.floor(Math.random() * 90_000_000) + 10_000_000, // 8-digit random
  };
  EDITOR_MODULES.forEach(({ key, fields }) => {
    const [vals] = editorMap[key];
    fields.forEach(({ id }) => { if (vals?.[id] !== undefined) event[id] = vals[id]; });
  });
  return event;
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

// ── PredictFooter: display below the sliders when the panel is open ──────────────────────
function PredictFooter({ editorMap, onPredict }) {
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);

  async function handlePredict() {
    setLoading(true);
    setError(null);
    try {
      const event = collectEvent(editorMap);
      const res  = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ event }),
      });
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.detail ?? `HTTP ${res.status}`);
      if (typeof onPredict === "function") onPredict(data.result);
    } catch (e) {
      setError(`Predict error: ${e.message}`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{
      padding: "10px 14px",
      borderTop: "0.5px solid var(--color-border-tertiary)",
      display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12,
    }}>
      <div style={{ fontSize: 10, color: "var(--color-text-tertiary)" }}>
        Params from all tabs — click Predict to run the model.
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
        {error && (
          <span style={{ fontSize: 11, color: "#c44" }}>{error}</span>
        )}
        {loading && (
          <span style={{ fontSize: 11, color: "var(--color-text-tertiary)" }}>
            <span style={{ display: "inline-block", animation: "spin 1s linear infinite" }}>⏳</span>
            {" "}Running prediction…
            <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
          </span>
        )}
        <button
          onClick={handlePredict}
          disabled={loading}
          style={{
            fontSize: 12, padding: "6px 16px", borderRadius: 6, border: "none",
            background: loading ? "var(--color-background-secondary)" : ACCENT,
            color: loading ? "var(--color-text-tertiary)" : "#fff",
            cursor: loading ? "default" : "pointer",
            fontWeight: 600, letterSpacing: "0.02em",
          }}
        >
          {loading ? "…" : "▶ Predict"}
        </button>
      </div>
    </div>
  );
}

// ── Main export ───────────────────────────────────────────────────────────────
export function ParamEditor({ editorMap, onPredict }) {
  const [open, setOpen] = useState(false);

  return (
    <div style={{ background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: 10, marginBottom: 14, overflow: "hidden" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "10px 16px" }}>
        <span style={{ fontSize: 13 }}>⚙</span>
        <span style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-primary)", flex: 1 }}>
          Adjust parameters
        </span>
        <span style={{ fontSize: 10, color: "var(--color-text-tertiary)", marginRight: 8 }}>
          Slider + numbers — real-time adjustment
        </span>
        <button
          onClick={() => setOpen((p) => !p)}
          style={{ background: "none", border: "none", cursor: "pointer", fontSize: 11, color: "var(--color-text-tertiary)", padding: 0 }}
        >
          {open ? "▲" : "▼"}
        </button>
      </div>

      {open && (
        <>
          <div style={{ borderTop: "0.5px solid var(--color-border-tertiary)" }}>
            {EDITOR_MODULES.map((mod) => {
              const [vals, setVals] = editorMap[mod.key];
              return <EditorModuleSection key={mod.key} mod={mod} vals={vals} onChange={setVals} />;
            })}
          </div>
          {/* Predict button footer — only shown when panel is open */}
          <PredictFooter editorMap={editorMap} onPredict={onPredict} />
        </>
      )}
    </div>
  );
}