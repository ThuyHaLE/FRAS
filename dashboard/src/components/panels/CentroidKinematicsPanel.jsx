// dashboard/src/components/panels/CentroidKinematicsPanel.jsx
// Panel for displaying centroid kinematics features and alarms

import { useState, useCallback } from "react";
import { MiniStat } from "../shared/MiniStat";
import { GroupCard } from "../shared/GroupCard";
import { AlarmBanner } from "../shared/AlarmBanner";
import { CK_FEATURES, CK_GROUPS } from "../../constants/featureRanges";
import { useCkAlarm } from "../../alarms/tabAlarms";

export function CentroidKinematicsPanel({ vals, setVals }) {
  const [active, setActive] = useState(null);
  const onToggle = useCallback((id) => setActive((p) => (p === id ? null : id)), []);
  const onUpdate = useCallback((id, v) => {
    setVals((prev) => {
      const next = { ...prev, [id]: v };
      if (id === "spread_bearing_deg") {
        next.spread_bearing_sin = parseFloat(Math.sin((v * Math.PI) / 180).toFixed(3));
        next.spread_bearing_cos = parseFloat(Math.cos((v * Math.PI) / 180).toFixed(3));
      }
      return next;
    });
  }, [setVals]);
  const spd      = vals["centroid_speed_m_per_h"];
  const dirs     = ["N","NE","E","SE","S","SW","W","NW"];
  const cardinal = dirs[Math.round(vals["spread_bearing_deg"] / 45) % 8];
  const advRadius = spd >= 1000 ? (spd / 1000).toFixed(1) + " km" : Math.round(spd) + " m";
  const ckAlarm  = useCkAlarm(vals);
  return (
    <div style={{ padding: "1rem 0" }}>
      <AlarmBanner title="Mobility signals" accentColor="#378ADD" {...ckAlarm} />
      <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0,1fr))", gap: 10, marginBottom: 14 }}>
        <MiniStat label="Est. advance radius (1h)" value={advRadius} sub="centroid_speed_m_per_h × 1h" />
        <MiniStat label="Threat direction" value={`${Math.round(vals["spread_bearing_deg"])}° ${cardinal}`} sub="spread_bearing_deg → cardinal" />
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0,1fr))", gap: 10 }}>
        {CK_GROUPS.map((g, gi) => (
          <GroupCard key={gi} group={g} features={CK_FEATURES.filter((f) => f.group === gi)} vals={vals} activeFeature={active} onToggle={onToggle} onUpdate={onUpdate} />
        ))}
      </div>
    </div>
  );
}