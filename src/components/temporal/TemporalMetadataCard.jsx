// src/components/temporal/TemporalMetadataCard.jsx
// This component displays temporal metadata (hour, day of week, month) for a given event.

import { R, ACCENT, DOW_LABELS, MONTH_LABELS } from "./constants/featureRanges";

export function TemporalMetadataCard({ vals }) {
  const hour       = vals.event_start_hour;
  const isPeakHeat = hour >= 11 && hour < 17;
  const isDaytime  = hour >= 6 && hour < 20;
  const hourLabel  = isPeakHeat ? "Peak heat" : isDaytime ? "Daytime" : "Nighttime";
  const dow        = vals.event_start_dayofweek;
  const month      = vals.event_start_month;
  const hourPct    = Math.round((hour / R("event_start_hour").max) * 100);
  return (
    <div style={{ background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: 10, padding: ".9rem 1rem" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 12, paddingBottom: 8, borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
        <div style={{ width: 9, height: 9, borderRadius: 2, flexShrink: 0, background: "#D85A30" }} />
        <div style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-primary)" }}>Temporal Metadata</div>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        <div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 4 }}>
            <div>
              <div style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>event_start_hour</div>
              <div style={{ fontSize: 10, color: "var(--color-text-tertiary)", marginTop: 1 }}>Giờ bắt đầu (0–23)</div>
            </div>
            <div style={{ textAlign: "right" }}>
              <span style={{ fontSize: 20, fontWeight: 500, color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}>{String(hour).padStart(2, "0")}:00</span>
              <span style={{ fontSize: 10, marginLeft: 6, padding: "1px 6px", borderRadius: 3, background: "var(--color-background-secondary)", color: "var(--color-text-tertiary)" }}>{hourLabel}</span>
            </div>
          </div>
          <div style={{ position: "relative", height: 6, background: "var(--color-background-secondary)", borderRadius: 3, overflow: "hidden" }}>
            <div style={{ position: "absolute", top: 0, left: `${hourPct}%`, transform: "translateX(-50%)", width: 3, height: "100%", background: ACCENT, borderRadius: 1 }} />
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "var(--color-text-tertiary)", marginTop: 2 }}>
            <span>00:00</span><span>06:00</span><span>12:00</span><span>18:00</span><span>23:00</span>
          </div>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <div style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>event_start_dayofweek</div>
            <div style={{ fontSize: 10, color: "var(--color-text-tertiary)", marginTop: 1 }}>0 = Mon, 6 = Sun</div>
          </div>
          <div style={{ display: "flex", gap: 3 }}>
            {DOW_LABELS.map((d, i) => (
              <div key={i} style={{ width: 24, height: 24, borderRadius: 4, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9, fontWeight: i === dow ? 600 : 400, background: i === dow ? ACCENT : "var(--color-background-secondary)", color: i === dow ? "#fff" : "var(--color-text-tertiary)" }}>{d}</div>
            ))}
          </div>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <div style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>event_start_month</div>
            <div style={{ fontSize: 10, color: "var(--color-text-tertiary)", marginTop: 1 }}>Tháng bắt đầu (1–12)</div>
          </div>
          <div style={{ display: "flex", gap: 2, flexWrap: "wrap", justifyContent: "flex-end", maxWidth: 200 }}>
            {MONTH_LABELS.map((m, i) => (
              <div key={i} style={{ width: 22, height: 18, borderRadius: 3, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 8, fontWeight: i + 1 === month ? 600 : 400, background: i + 1 === month ? ACCENT : "var(--color-background-secondary)", color: i + 1 === month ? "#fff" : "var(--color-text-tertiary)" }}>{m}</div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}