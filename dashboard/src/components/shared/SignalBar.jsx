// dashboard/src/components/shared/SignalBar.jsx
// A simple horizontal bar that fills according to a value between 0 and 1, colored according to the alarm level.

import { clamp01 } from "../../utils/alarmUtils";

export function SignalBar({ value, color }) {
  const pct = Math.round(clamp01(value) * 100);
  return (
    <div style={{ height: 4, borderRadius: 2, background: "var(--color-background-secondary)", overflow: "hidden" }}>
      <div style={{ height: "100%", borderRadius: 2, width: `${pct}%`, background: color, transition: "width .3s" }} />
    </div>
  );
}