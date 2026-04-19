// src/FireRiskDashboard.jsx
// Main dashboard component for the Fire Risk Assessment System (FRAS).

import React, { useState, useCallback, useEffect } from "react";
import { fetchAllModules } from "./data/mockData";
import { THEMES } from "./constants/featureRanges";
import { CentroidKinematicsPanel } from "./components/panels/CentroidKinematicsPanel";
import { FireGrowthPanel }         from "./components/panels/FireGrowthPanel";
import { DirectionalityPanel }     from "./components/panels/DirectionalityPanel";
import { ProximityPanel }          from "./components/panels/ProximityPanel";
import { RiskHorizonPanel }        from "./components/panels/RiskHorizonPanel";
import { OverviewPanel }           from "./components/panels/OverviewPanel";
import { JsonUploadModal }         from "./components/editor/JsonUploadModal";

// ═══════════════════════════════════════════════════════════════════════════════
// ── MAIN DASHBOARD
// ═══════════════════════════════════════════════════════════════════════════════
const TABS = [
  { id: 0, label: "Overview",             icon: "◈" },
  { id: 1, label: "Centroid Kinematics",  icon: "⊙" },
  { id: 2, label: "Growth Features",      icon: "△" },
  { id: 3, label: "Directionality",       icon: "→" },
  { id: 4, label: "Distance to Evac Zone Centroids", icon: "⬡" },
  { id: 5, label: "Risk Horizon",         icon: "◷" },
];

function ThemeToggle({ theme, setTheme }) {
  return (
    <button onClick={() => setTheme(theme === "dark" ? "light" : "dark")} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10, padding: "3px 10px", borderRadius: 4, border: "0.5px solid var(--color-border-secondary)", background: "transparent", color: "var(--color-text-tertiary)", cursor: "pointer" }}>
      {theme === "dark" ? "☀ Light" : "☾ Dark"}
    </button>
  );
}

export default function FireRiskDashboard() {
  const [activeTab, setActiveTab] = useState(0);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState(null);
  const [theme, setTheme]         = useState("light");
  const [eventId, setEventId]     = useState(null);
  const [ckVals,  setCkVals]      = useState(null);
  const [fgVals,  setFgVals]      = useState(null);
  const [dirVals, setDirVals]     = useState(null);
  const [rsVals,  setRsVals]      = useState(null);
  const [rhVals,  setRhVals]      = useState(null);
  const [tcVals,  setTcVals]      = useState(null);
  const [tmVals,  setTmVals]      = useState(null);

  const loadData = useCallback(async () => {
    try {
      setLoading(true); setError(null);
      const data = await fetchAllModules();
      setEventId(data.eventId); setCkVals(data.centroidKinematics); setFgVals(data.fireGrowth);
      setDirVals(data.directionality); setRsVals(data.riskScore); setRhVals(data.reachProbability);
      setTcVals(data.temporalCoverage); setTmVals(data.temporalMetadata);
    } catch (e) { setError(e.message || "Failed to load data"); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  if (loading) return <div style={{ padding: "2rem", color: "var(--color-text-tertiary)", fontSize: 13 }}>Đang tải dữ liệu…</div>;
  if (error || !ckVals) return (
    <div style={{ padding: "2rem", color: "#c44", fontSize: 13 }}>
      ⚠ {error || "No data"} — <span style={{ textDecoration: "underline", cursor: "pointer" }} onClick={loadData}>Thử lại</span>
    </div>
  );

  return (
    <div style={{ fontFamily: "var(--font-sans, system-ui, sans-serif)", padding: "1rem", boxSizing: "border-box", background: THEMES[theme]["--color-background-tertiary"], minHeight: "100vh", ...THEMES[theme] }}>
      <div style={{ marginBottom: 16, paddingBottom: 14, borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ fontSize: 18, fontWeight: 600, color: "var(--color-text-primary)", letterSpacing: "-0.01em" }}>
            🔥 FRAS — Fire Risk Assessment System
          </div>
          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            {eventId && <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--color-text-tertiary)" }}>event_id: {eventId}</span>}
            <ThemeToggle theme={theme} setTheme={setTheme} />
            <button onClick={loadData} style={{ fontSize: 10, padding: "3px 10px", borderRadius: 4, border: "0.5px solid var(--color-border-secondary)", background: "transparent", color: "var(--color-text-tertiary)", cursor: "pointer" }}>↻ Refresh</button>
          </div>
        </div>
      </div>
      <div style={{ display: "flex", gap: 4, marginBottom: 16, overflowX: "auto", paddingBottom: 4 }}>
        {TABS.map((tab) => {
          const isActive = activeTab === tab.id;
          return (
            <button key={tab.id} onClick={() => setActiveTab(tab.id)} style={{ display: "flex", alignItems: "center", gap: 6, flexShrink: 0, padding: "7px 14px", borderRadius: 7, border: "none", cursor: "pointer", fontSize: 12, fontWeight: isActive ? 600 : 400, transition: "all .15s", background: isActive ? "var(--color-background-primary)" : "transparent", color: isActive ? "var(--color-text-primary)" : "var(--color-text-secondary)", boxShadow: isActive ? `0 0 0 0.5px var(--color-border-secondary)` : "none" }}>
              <span style={{ fontSize: 13 }}>{tab.icon}</span>
              {tab.label}
            </button>
          );
        })}
      </div>
      <div>
        {activeTab === 0 && <OverviewPanel ckVals={ckVals} setCkVals={setCkVals} fgVals={fgVals} setFgVals={setFgVals} dirVals={dirVals} setDirVals={setDirVals} rsVals={rsVals} setRsVals={setRsVals} rhVals={rhVals} tcVals={tcVals} setTcVals={setTcVals} tmVals={tmVals} setTmVals={setTmVals} eventId={eventId} onGoTab={setActiveTab} />}
        {activeTab === 1 && <CentroidKinematicsPanel vals={ckVals} setVals={setCkVals} />}
        {activeTab === 2 && <FireGrowthPanel vals={fgVals} setVals={setFgVals} />}
        {activeTab === 3 && <DirectionalityPanel vals={dirVals} setVals={setDirVals} />}
        {activeTab === 4 && <ProximityPanel vals={rsVals} setVals={setRsVals} />}
        {activeTab === 5 && rhVals && <RiskHorizonPanel vals={rhVals} rsVals={rsVals} />}
      </div>
    </div>
  );
}