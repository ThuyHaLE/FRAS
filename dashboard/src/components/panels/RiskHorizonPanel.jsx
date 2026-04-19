// dashboard/src/components/panels/RiskHorizonPanel.jsx
// This panel shows the probability of reaching the nearest evacuation zone 
// within different time horizons (e.g. 6h, 12h, 24h, 72h).

import { RH_HORIZONS, ACCENT } from "../../constants/featureRanges";

function HorizonCard({ horizon, prob }) {
  const pct = Math.round(prob * 100);
  return (
    <div style={{ borderRadius: 10, padding: "14px 16px", border: "0.5px solid var(--color-border-tertiary)", background: "var(--color-background-primary)" }}>
      <div style={{ fontSize: 10, color: "var(--color-text-tertiary)", marginBottom: 6 }}>{horizon.label}</div>
      <div style={{ fontSize: 32, fontWeight: 500, lineHeight: 1, color: "var(--color-text-primary)", fontFamily: "var(--font-mono)", marginBottom: 10 }}>{pct}%</div>
      <div style={{ height: 4, borderRadius: 2, background: "var(--color-background-secondary)", overflow: "hidden" }}>
        <div style={{ height: "100%", borderRadius: 2, width: `${pct}%`, background: ACCENT, transition: "width .3s" }} />
      </div>
      <div style={{ fontSize: 10, color: "var(--color-text-tertiary)", marginTop: 6, fontFamily: "var(--font-mono)" }}>{horizon.short}</div>
    </div>
  );
}

function TimelineBar({ probs }) {
  const points = RH_HORIZONS.map((h, i) => ({ ...h, prob: probs[h.key], x: 10 + (i / (RH_HORIZONS.length - 1)) * 80 }));
  return (
    <div style={{ position: "relative", height: 72, margin: "8px 0 4px" }}>
      <div style={{ position: "absolute", top: 28, left: "10%", right: "10%", height: 2, background: ACCENT, borderRadius: 1, opacity: 0.3 }} />
      {points.map((p) => (
        <div key={p.key} style={{ position: "absolute", top: 0, left: `${p.x}%`, transform: "translateX(-50%)", display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
          <div style={{ fontSize: 11, fontWeight: 500, color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}>{Math.round(p.prob * 100)}%</div>
          <div style={{ width: 28, height: 28, borderRadius: "50%", background: "var(--color-background-secondary)", border: `2px solid ${ACCENT}`, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <div style={{ width: 8, height: 8, borderRadius: "50%", background: ACCENT }} />
          </div>
          <div style={{ fontSize: 10, color: "var(--color-text-tertiary)" }}>{p.short}</div>
        </div>
      ))}
    </div>
  );
}

export function RiskHorizonPanel({ vals, rsVals }) {
  const distMin = rsVals?.dist_min_ci_0_5h ?? 0;
  const closingSpeed = rsVals?.closing_speed_m_per_h ?? 0;
  const etaH = closingSpeed > 5 ? (distMin / closingSpeed) : null;
  return (
    <div style={{ padding: "1rem 0" }}>
      <div style={{ background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: 10, padding: "1rem 1.25rem", marginBottom: 14 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12, paddingBottom: 10, borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
          <div>
            <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", textTransform: "uppercase", letterSpacing: ".07em", marginBottom: 2 }}>Evacuation zone reach probability</div>
            <div style={{ fontSize: 13, fontWeight: 500, color: "var(--color-text-primary)" }}>Risk horizon — nearest evac zone</div>
          </div>
          <div style={{ textAlign: "right" }}>
            <div style={{ fontSize: 10, color: "var(--color-text-tertiary)", marginBottom: 3 }}>current distance</div>
            <div style={{ fontSize: 16, fontWeight: 500, color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}>
              {distMin >= 1000 ? (distMin / 1000).toFixed(2) + " km" : distMin + " m"}
            </div>
            {etaH && <div style={{ fontSize: 10, color: "var(--color-text-tertiary)", marginTop: 2 }}>ETA ~{etaH < 1 ? Math.round(etaH * 60) + " min" : etaH.toFixed(1) + " h"} @ {closingSpeed} m/h</div>}
          </div>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0,1fr))", gap: 10, marginBottom: 16 }}>
          {RH_HORIZONS.map((h) => <HorizonCard key={h.key} horizon={h} prob={vals[h.key]} />)}
        </div>
        <div style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-secondary)", marginBottom: 4 }}>Progression timeline</div>
        <TimelineBar probs={vals} />
        <div style={{ display: "flex", justifyContent: "space-between", padding: "0 10%", marginTop: 2 }}>
          <span style={{ fontSize: 10, color: "var(--color-text-tertiary)" }}>Now</span>
          <span style={{ fontSize: 10, color: "var(--color-text-tertiary)" }}>72 h</span>
        </div>
      </div>
    </div>
  );
}