// dashboard/src/components/shared/CrossModuleAlarmCard.jsx
// A compact card component to display alarm information across different modules (e.g. temporal, spatial, etc.)

import { clamp01, alarmLevel, BADGE_STYLES, ALARM_DOT_COLORS } from "../../utils/alarmUtils";
import { SignalBar } from "./SignalBar";

function SourceTag({ label }) {
  return (
    <span style={{ fontSize: 9, padding: "1px 6px", borderRadius: 3, fontFamily: "var(--font-mono)", background: "var(--color-background-tertiary)", color: "var(--color-text-tertiary)" }}>
      {label}
    </span>
  );
}

export function CrossModuleAlarmCard({ scenario }) {
  const { title, subtitle, sources, accentColor, overallScore, signals, summary } = scenario;
  const level    = alarmLevel(overallScore);
  const badge    = BADGE_STYLES[level];
  const dotColor = ALARM_DOT_COLORS[level];
  return (
    <div style={{ background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: 10, overflow: "hidden", marginBottom: 12 }}>
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", padding: "10px 14px", borderBottom: "0.5px solid var(--color-border-tertiary)", gap: 8 }}>
        <div>
          <div style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-primary)" }}>{title}</div>
          <div style={{ fontSize: 10, color: "var(--color-text-tertiary)", marginTop: 2, lineHeight: 1.4 }}>{subtitle}</div>
        </div>
        <span style={{ fontSize: 10, fontWeight: 500, padding: "2px 9px", borderRadius: 4, letterSpacing: ".04em", flexShrink: 0, background: badge.bg, color: badge.color }}>{badge.label}</span>
      </div>
      <div style={{ display: "flex", gap: 4, flexWrap: "wrap", padding: "6px 14px", borderBottom: "0.5px solid var(--color-border-tertiary)", background: "var(--color-background-secondary)" }}>
        {sources.map((s) => <SourceTag key={s} label={s} />)}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: `repeat(${signals.length}, minmax(0,1fr))`, gap: 1, background: "var(--color-border-tertiary)" }}>
        {signals.map((s) => (
          <div key={s.name} style={{ background: "var(--color-background-primary)", padding: "10px 12px", display: "flex", flexDirection: "column", gap: 5 }}>
            <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
              <span style={{ fontSize: 10, color: "var(--color-text-tertiary)", fontFamily: "var(--font-mono)" }}>{s.name}</span>
              <span style={{ fontSize: 13, fontWeight: 500, fontFamily: "var(--font-mono)", color: "var(--color-text-primary)" }}>{Math.round(clamp01(s.value) * 100)}%</span>
            </div>
            <SignalBar value={s.value} color={accentColor} />
            <div style={{ fontSize: 10, color: "var(--color-text-tertiary)", lineHeight: 1.4 }}>{s.hint}</div>
          </div>
        ))}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 14px", borderTop: "0.5px solid var(--color-border-tertiary)", background: "var(--color-background-secondary)" }}>
        <div style={{ width: 7, height: 7, borderRadius: "50%", flexShrink: 0, background: dotColor }} />
        <div style={{ fontSize: 11, color: "var(--color-text-secondary)", lineHeight: 1.5 }}>{summary}</div>
      </div>
    </div>
  );
}