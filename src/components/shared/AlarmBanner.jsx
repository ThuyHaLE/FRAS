// src/components/shared/AlarmBanner.jsx
// A compact banner component to display alarm signals and summary information.

import { clamp01, alarmLevel, BADGE_STYLES, ALARM_DOT_COLORS } from "./utils/alarmUtils";
import { SignalBar } from "./SignalBar";

export function AlarmBanner({ title, signals, summary, overallScore, accentColor }) {
  const level    = alarmLevel(overallScore);
  const badge    = BADGE_STYLES[level];
  const dotColor = ALARM_DOT_COLORS[level];
  return (
    <div style={{ borderRadius: 10, border: "0.5px solid var(--color-border-tertiary)", background: "var(--color-background-primary)", overflow: "hidden", marginBottom: 14 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 14px", borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
        <span style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-primary)" }}>{title}</span>
        <span style={{ fontSize: 10, fontWeight: 500, padding: "2px 9px", borderRadius: 4, letterSpacing: ".04em", background: badge.bg, color: badge.color }}>{badge.label}</span>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: `repeat(${signals.length}, minmax(0,1fr))`, gap: 1, background: "var(--color-border-tertiary)" }}>
        {signals.map((s) => (
          <div key={s.name} style={{ background: "var(--color-background-primary)", padding: "10px 12px", display: "flex", flexDirection: "column", gap: 6 }}>
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