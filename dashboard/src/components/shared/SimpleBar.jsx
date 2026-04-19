// dashboard/src/components/shared/SimpleBar.jsx
// A simple horizontal bar that fills according to the normalized value of a feature. Used in the metadata cards.

import { ACCENT } from "../../constants/featureRanges";

export function SimpleBar({ feature, value }) {
  const norm = Math.min(Math.max(feature.toNorm(value), 0), 1);
  const pct = Math.round(norm * 100);
  return (
    <div style={{ position: "relative", height: 6, background: "var(--color-background-secondary)", borderRadius: 3, overflow: "hidden", marginBottom: 2 }}>
      <div style={{ position: "absolute", top: 0, height: "100%", borderRadius: 3, transition: "width .25s", background: ACCENT, width: `${pct}%`, opacity: 0.7 }} />
    </div>
  );
}