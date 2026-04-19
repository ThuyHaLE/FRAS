// src/components/temporal/TemporalCoverageCard.jsx
// This component displays temporal coverage metrics for a fire event, 
// including the number of perimeters in the first 5 hours and the time between the first and last perimeter. 
// It also indicates whether the temporal resolution is considered low based on specific criteria.

export function TemporalCoverageCard({ vals }) {
  const isLowRes = vals.low_temporal_resolution_0_5h === 1;
  return (
    <div style={{ background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: 10, padding: ".9rem 1rem" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 12, paddingBottom: 8, borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
        <div style={{ width: 9, height: 9, borderRadius: 2, flexShrink: 0, background: "#7F77DD" }} />
        <div style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-primary)" }}>Temporal Coverage</div>
        <span style={{ marginLeft: "auto", fontSize: 11, fontWeight: 500, padding: "1px 7px", borderRadius: 3, background: "var(--color-background-secondary)", color: "var(--color-text-secondary)" }}>
          {isLowRes ? "LOW RES" : "OK"}
        </span>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {[
          { label: "num_perimeters_0_5h", desc: "Số perimeter trong 5h đầu",          val: vals.num_perimeters_0_5h,  fmt: (v) => v },
          { label: "dt_first_last_0_5h",  desc: "Khoảng thời gian first–last perimeter", val: vals.dt_first_last_0_5h, fmt: (v) => v.toFixed(1) + " h" },
        ].map(({ label, desc, val, fmt }) => (
          <div key={label} style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <div style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>{label}</div>
              <div style={{ fontSize: 10, color: "var(--color-text-tertiary)", marginTop: 1 }}>{desc}</div>
            </div>
            <div style={{ fontSize: 20, fontWeight: 500, color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}>{fmt(val)}</div>
          </div>
        ))}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <div style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>low_temporal_resolution_0_5h</div>
            <div style={{ fontSize: 10, color: "var(--color-text-tertiary)", marginTop: 1 }}>Flag: dt &lt; 0.5h hoặc chỉ 1 perimeter</div>
          </div>
          <span style={{ fontSize: 11, fontWeight: 500, padding: "2px 8px", borderRadius: 3, background: "var(--color-background-secondary)", color: "var(--color-text-secondary)", fontFamily: "var(--font-mono)" }}>
            {vals.low_temporal_resolution_0_5h} — {isLowRes ? "YES" : "NO"}
          </span>
        </div>
      </div>
    </div>
  );
}