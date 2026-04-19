// dashboard/src/components/panels/ProximityPanel.jsx
// Panel for proximity-related features and alarms in the FRAS dashboard.

import { useState, useCallback } from "react";
import { MiniStat } from "../shared/MiniStat";
import { GroupCard } from "../shared/GroupCard";
import { AlarmBanner } from "../shared/AlarmBanner";
import { RS_GROUPS, RS_FEATURES } from "../../constants/featureRanges";
import { useRsAlarm } from "../../alarms/tabAlarms";

export function ProximityPanel({ vals, setVals }) {
  const [active, setActive] = useState(null);
  const onToggle = useCallback((id) => setActive((p) => (p === id ? null : id)), []);
  const onUpdate = useCallback((id, v) => setVals((prev) => ({ ...prev, [id]: v })), [setVals]);

  const cs = vals["closing_speed_m_per_h"];
  const dm = vals["dist_min_ci_0_5h"];
  const etaH = cs > 5 ? Math.round(dm / cs) : null;
  const etaText = etaH !== null ? `~${etaH} h` : "N/A";
  const r2 = vals["dist_fit_r2_0_5h"];
  const rsAlarm = useRsAlarm(vals);

  return (
    <div style={{ padding: "1rem 0" }}>
      <AlarmBanner title="Proximity signals" accentColor="#5B8DD9" {...rsAlarm} />

      <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0,1fr))", gap: 10, marginBottom: 14 }}>
        <MiniStat label="ETA to zone edge" value={etaText} sub="dist_min ÷ closing_speed" />
        <MiniStat label="Fit confidence (R²)" value={r2.toFixed(2)} sub="trajectory fit quality" />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0,1fr))", gap: 10 }}>
        {RS_GROUPS.map((g, gi) => (
          <GroupCard key={gi} group={g} features={RS_FEATURES.filter((f) => f.group === gi)} vals={vals} activeFeature={active} onToggle={onToggle} onUpdate={onUpdate} />
        ))}
      </div>
    </div>
  );
}