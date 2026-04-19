// dashboard/src/components/shared/FeatureCard.jsx
// A card component for displaying a feature with its value, 
// and allowing the user to toggle it active/inactive and adjust its value if active.

import { FEATURE_RANGES, ACCENT } from "../../constants/featureRanges";
import { fmtVal, formatTick } from "../../utils/alarmUtils";
import { SimpleBar } from "./SimpleBar";

export function FeatureCard({ feature, value, isActive, onToggle, onUpdate }) {
  const rng = FEATURE_RANGES[feature.id];
  return (
    <div
      onClick={() => onToggle(feature.id)}
      style={{ marginBottom: 8, padding: "6px 8px", borderRadius: 6, cursor: "pointer", border: `0.5px solid ${isActive ? "var(--color-border-secondary)" : "transparent"}`, background: isActive ? "var(--color-background-secondary)" : "transparent", transition: "background .15s" }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 3 }}>
        <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>{feature.label}</span>
        <span style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}>{fmtVal(feature, value, FEATURE_RANGES)}</span>
      </div>
      <SimpleBar feature={feature} value={value} />
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "var(--color-text-tertiary)", marginTop: 2 }}>
        {(feature.scaleLabels || []).map((l, i) => <span key={i}>{formatTick(l)}</span>)}
      </div>
      {isActive && rng && (
        <div onClick={(e) => e.stopPropagation()} style={{ marginTop: 8, paddingTop: 8, borderTop: "0.5px solid var(--color-border-tertiary)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <input type="range" min={rng.min} max={rng.max} step={rng.step} value={value} onChange={(e) => onUpdate(feature.id, +e.target.value)} style={{ flex: 1, accentColor: ACCENT }} />
            <span style={{ fontSize: 11, fontWeight: 500, minWidth: 60, textAlign: "right", fontFamily: "var(--font-mono)", color: "var(--color-text-primary)" }}>{fmtVal(feature, value, FEATURE_RANGES)}</span>
          </div>
          {feature.hint && <div style={{ fontSize: 10, color: "var(--color-text-tertiary)", lineHeight: 1.5, marginTop: 4, fontStyle: "italic" }}>{feature.hint}</div>}
        </div>
      )}
    </div>
  );
}