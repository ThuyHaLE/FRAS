// dashboard/src/components/panels/OverviewPanel.jsx
// This component serves as a high-level overview of the current event, 
// showing key parameters and metrics across modules, as well as temporal coverage and metadata. 
// It also includes the new cross-module alarms summary.

import { ParamEditor } from "../editor/ParamEditor";
import { TemporalCoverageCard } from "../temporal/TemporalCoverageCard";
import { TemporalMetadataCard } from "../temporal/TemporalMetadataCard";
import { CrossModuleAlarms } from "../../alarms/tabAlarms";
import { ACCENT, RH_HORIZONS } from "../../constants/featureRanges";

function useEditorVals({ ckVals, setCkVals, fgVals, setFgVals, dirVals, setDirVals, rsVals, setRsVals, tcVals, setTcVals, tmVals, setTmVals }) {
  return { ck: [ckVals, setCkVals], fg: [fgVals, setFgVals], dir: [dirVals, setDirVals], rs: [rsVals, setRsVals], tc: [tcVals, setTcVals], tm: [tmVals, setTmVals] };
}

export function OverviewPanel({ ckVals, setCkVals, fgVals, setFgVals, dirVals, setDirVals, rsVals, setRsVals, rhVals, tcVals, setTcVals, tmVals, setTmVals, eventId, onGoTab }) {
  const editorMap = useEditorVals({ ckVals, setCkVals, fgVals, setFgVals, dirVals, setDirVals, rsVals, setRsVals, tcVals, setTcVals, tmVals, setTmVals });

  return (
    <div style={{ padding: "1rem 0" }}>
      <ParamEditor editorMap={editorMap} />
      {/* reach probability*/}
      {rhVals && (
        <div onClick={() => onGoTab(5)} style={{ background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: 10, padding: "12px 16px", marginBottom: 14, cursor: "pointer" }}>
          <div style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-primary)", marginBottom: 10 }}>Reach probability — evac zone</div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0,1fr))", gap: 8 }}>
            {RH_HORIZONS.map((h) => {
              const pct = Math.round(rhVals[h.key] * 100);
              return (
                <div key={h.key} style={{ textAlign: "center" }}>
                  <div style={{ fontSize: 10, color: "var(--color-text-tertiary)", marginBottom: 4 }}>{h.short}</div>
                  <div style={{ fontSize: 18, fontWeight: 500, color: "var(--color-text-primary)", lineHeight: 1, marginBottom: 4, fontFamily: "var(--font-mono)" }}>{pct}%</div>
                  <div style={{ height: 3, borderRadius: 2, background: "var(--color-background-secondary)", overflow: "hidden" }}>
                    <div style={{ height: "100%", width: `${pct}%`, background: ACCENT, borderRadius: 2 }} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
      {/* ── CROSS-MODULE ALARMS — thêm mới ── */}
      {ckVals && fgVals && dirVals && rsVals && tcVals && tmVals && rhVals && (
        <CrossModuleAlarms
          ckVals={ckVals} fgVals={fgVals} dirVals={dirVals}
          rsVals={rsVals} tcVals={tcVals} tmVals={tmVals} rhVals={rhVals}
        />
      )}
      {/* temporal cards*/}
      {(tcVals || tmVals) && (
      <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 14 }}>
          {tcVals && <TemporalCoverageCard vals={tcVals} />}
          {tmVals && <TemporalMetadataCard vals={tmVals} />}
      </div>
      )}
    </div>
  );
}
