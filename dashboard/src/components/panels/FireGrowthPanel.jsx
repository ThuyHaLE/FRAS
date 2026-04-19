// dashboard/src/components/panels/FireGrowthPanel.jsx
// Panel for fire growth related features and alarms

import { useState, useCallback } from "react";
import { applyFGDerived, useFgAlarm } from "../../utils/alarmUtils";
import { FG_FEATURES, FG_GROUPS } from "../../constants/featureRanges";
import { MiniStat } from "../shared/MiniStat";
import { GroupCard } from "../shared/GroupCard";
import { AlarmBanner } from "../shared/AlarmBanner";

export function FireGrowthPanel({ vals, setVals }) {
  const [active, setActive] = useState(null);
  const onToggle = useCallback((id) => setActive((p) => (p === id ? null : id)), []);
  const onUpdate = useCallback((id, v) => setVals((prev) => applyFGDerived(prev, id, v)), [setVals]);

  const rate = vals["area_growth_rate_ha_per_h"];
  const radial = vals["radial_growth_rate_m_per_h"];
  const fgAlarm = useFgAlarm(vals);

  return (
    <div style={{ padding: "1rem 0" }}>
      <AlarmBanner title="Growth signals" accentColor="#D85A30" {...fgAlarm} />
      <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0,1fr))", gap: 10, marginBottom: 14 }}>
        <MiniStat label="Growth rate" value={`${rate.toFixed(1)} ha/h`} sub="area_growth_rate_ha_per_h" />
        <MiniStat label="Radial spread rate" value={`${Math.round(radial)} m/h`} sub="radial_growth_rate_m_per_h" />
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0,1fr))", gap: 10 }}>
        {FG_GROUPS.map((g, gi) => (
          <GroupCard key={gi} group={g} features={FG_FEATURES.filter((f) => f.group === gi)} vals={vals} activeFeature={active} onToggle={onToggle} onUpdate={onUpdate} />
        ))}
      </div>
    </div>
  );
}