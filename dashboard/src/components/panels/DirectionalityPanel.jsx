// src/components/panels/DirectionalityPanel.jsx
// This panel focuses on features related to the directionality of the threat, 
// such as alignment cosine and along-track speed, 
// which indicate how directly the threat is moving towards the evacuation zone. 
// It also includes an alarm banner to highlight any significant directional threat signals.

import { useState, useCallback } from "react";
import { MiniStat } from "./components/shared/MiniStat";
import { GroupCard } from "./components/shared/GroupCard";
import { AlarmBanner } from "./components/shared/AlarmBanner";
import { DIR_FEATURES, DIR_GROUPS } from "./constants/featureRanges";
import { useDirAlarm } from "./alarms/tabAlarms";

export function DirectionalityPanel({ vals, setVals }) {
  const [active, setActive] = useState(null);
  const onToggle = useCallback((id) => setActive((p) => (p === id ? null : id)), []);
  const onUpdate = useCallback((id, v) => setVals((prev) => ({ ...prev, [id]: v })), [setVals]);

  const alignCos = vals["alignment_cos"];
  const alongSpeed = vals["along_track_speed"];
  const threat = alignCos > 0.7 ? "Direct" : alignCos > 0.3 ? "Oblique" : "Tangential";
  const dirAlarm = useDirAlarm(vals);

  return (
    <div style={{ padding: "1rem 0" }}>
      <AlarmBanner title="Directional threat signals" accentColor="#7F77DD" {...dirAlarm} />
      <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0,1fr))", gap: 10, marginBottom: 14 }}> 
        <MiniStat label="Threat type" value={threat} sub={`alignment_cos = ${alignCos.toFixed(2)}`} />
        <MiniStat label="Along-track speed" value={`${alongSpeed > 0 ? "+" : ""}${Math.round(alongSpeed)} m/h`} sub="toward evac zone" />
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0,1fr))", gap: 10 }}>
        {DIR_GROUPS.map((g) => (
          <GroupCard key={g.id} group={g} features={DIR_FEATURES.filter((f) => f.group === g.id)} vals={vals} activeFeature={active} onToggle={onToggle} onUpdate={onUpdate} />
        ))}
      </div>
    </div>
  );
}