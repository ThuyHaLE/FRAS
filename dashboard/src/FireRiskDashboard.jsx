// dashboard/src/FireRiskDashboard.jsx
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

const TABS = [
  { id: 0, label: "Overview",                        icon: "◈" },
  { id: 1, label: "Centroid Kinematics",             icon: "⊙" },
  { id: 2, label: "Growth Features",                 icon: "△" },
  { id: 3, label: "Directionality",                  icon: "→" },
  { id: 4, label: "Distance to Evac Zone Centroids", icon: "⬡" },
  { id: 5, label: "Risk Horizon",                    icon: "◷" },
];

function ThemeToggle({ theme, setTheme }) {
  return (
    <button
      onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
      style={{
        display: "flex", alignItems: "center", gap: 6,
        fontSize: 10, padding: "3px 10px", borderRadius: 4,
        border: "0.5px solid var(--color-border-secondary)",
        background: "transparent", color: "var(--color-text-tertiary)", cursor: "pointer",
      }}
    >
      {theme === "dark" ? "☀ Light" : "☾ Dark"}
    </button>
  );
}

// Mini banner display 4 prob_ after prediction
function PredictionBanner({ result, onDismiss }) {
  if (!result) return null;
  const PROB_KEYS = [
    { key: "prob_12h", label: "12h" },
    { key: "prob_24h", label: "24h" },
    { key: "prob_48h", label: "48h" },
    { key: "prob_72h", label: "72h" },
  ];
  return (
    <div style={{
      margin: "0 0 14px", padding: "10px 14px", borderRadius: 8,
      border: "0.5px solid #3a7d44", background: "rgba(58,125,68,0.08)",
      display: "flex", alignItems: "center", gap: 14,
    }}>
      <span style={{ fontSize: 15 }}>✅</span>
      <div style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-primary)", flexShrink: 0 }}>
        Predict successfully
      </div>
      <div style={{ display: "flex", gap: 16, flex: 1 }}>
        {PROB_KEYS.map(({ key, label }) => (
          <span key={key} style={{ fontSize: 12, fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>
            <span style={{ color: "var(--color-text-tertiary)" }}>{label}: </span>
            <span style={{ fontWeight: 600, color: "var(--color-text-primary)" }}>
              {result[key] != null ? `${(result[key] * 100).toFixed(1)}%` : "—"}
            </span>
          </span>
        ))}
      </div>
      <span style={{ fontSize: 10, color: "var(--color-text-tertiary)", flexShrink: 0 }}>
        event {result.event_id}
      </span>
      <button
        onClick={onDismiss}
        style={{ background: "none", border: "none", cursor: "pointer", fontSize: 14, color: "var(--color-text-tertiary)", lineHeight: 1 }}
      >
        ✕
      </button>
    </div>
  );
}

export default function FireRiskDashboard() {
  const [activeTab, setActiveTab] = useState(0);
  const [loading,   setLoading]   = useState(true);
  const [error,     setError]     = useState(null);
  const [theme,     setTheme]     = useState("light");
  const [eventId,   setEventId]   = useState(null);
  const [ckVals,    setCkVals]    = useState(null);
  const [fgVals,    setFgVals]    = useState(null);
  const [dirVals,   setDirVals]   = useState(null);
  const [rsVals,    setRsVals]    = useState(null);
  const [rhVals,    setRhVals]    = useState(null);
  const [tcVals,    setTcVals]    = useState(null);
  const [tmVals,    setTmVals]    = useState(null);

  const [showUpload,       setShowUpload]       = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);

  const loadData = useCallback(async () => {
    try {
      setLoading(true); setError(null);
      const data = await fetchAllModules();
      setEventId(data.eventId);
      setCkVals(data.centroidKinematics);
      setFgVals(data.fireGrowth);
      setDirVals(data.directionality);
      setRsVals(data.riskScore);
      setRhVals(data.reachProbability);
      setTcVals(data.temporalCoverage);
      setTmVals(data.temporalMetadata);
      setPredictionResult(null);
    } catch (e) {
      setError(e.message || "Failed to load data");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  // submission shape: { event_id, prob_12h, prob_24h, prob_48h, prob_72h }
  function handlePredictionResult(result) {
    setPredictionResult(result);

    // Update rhVals to RiskHorizonPanel render with new prob_ values after prediction
    setRhVals(prev => ({
      ...prev,
      prob_12h: result.prob_12h ?? prev?.prob_12h,
      prob_24h: result.prob_24h ?? prev?.prob_24h,
      prob_48h: result.prob_48h ?? prev?.prob_48h,
      prob_72h: result.prob_72h ?? prev?.prob_72h,
    }));

    // Update event_id on the header
    if (result.event_id != null) setEventId(result.event_id);

  }

  function handleUploadApply(features, result) {
    handlePredictionResult(result);

    // Mỗi setter chỉ nhận các key nó đang giữ, spread features vào
    setCkVals(prev => ({ ...prev, ...features }));
    setFgVals(prev => ({ ...prev, ...features }));
    setDirVals(prev => ({ ...prev, ...features }));
    setRsVals(prev => ({ ...prev, ...features }));
    setTcVals(prev => ({ ...prev, ...features }));
    setTmVals(prev => ({ ...prev, ...features }));
  }

  if (loading) return (
    <div style={{ padding: "2rem", color: "var(--color-text-tertiary)", fontSize: 13 }}>
      Uploading data… 
    </div>
  );
  if (error || !ckVals) return (
    <div style={{ padding: "2rem", color: "#c44", fontSize: 13 }}>
      ⚠ {error || "No data"} —{" "}
      <span style={{ textDecoration: "underline", cursor: "pointer" }} onClick={loadData}>Try again</span>
    </div>
  );

  return (
    <div style={{
      fontFamily: "var(--font-sans, system-ui, sans-serif)",
      padding: "1rem", boxSizing: "border-box",
      background: THEMES[theme]["--color-background-tertiary"],
      minHeight: "100vh",
      ...THEMES[theme],
    }}>
      {/* Header */}
      <div style={{ marginBottom: 16, paddingBottom: 14, borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ fontSize: 18, fontWeight: 600, color: "var(--color-text-primary)", letterSpacing: "-0.01em" }}>
            🔥 FRAS — Fire Risk Assessment System
          </div>
          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            {eventId && (
              <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--color-text-tertiary)" }}>
                event_id: {eventId}
              </span>
            )}
            <ThemeToggle theme={theme} setTheme={setTheme} />
            <button
              onClick={() => setShowUpload(true)}
              style={{
                fontSize: 10, padding: "3px 10px", borderRadius: 4,
                border: "0.5px solid var(--color-border-secondary)",
                background: "transparent", color: "var(--color-text-tertiary)", cursor: "pointer",
              }}
            >
              ⬆ Upload &amp; Predict
            </button>
            <button
              onClick={loadData}
              style={{
                fontSize: 10, padding: "3px 10px", borderRadius: 4,
                border: "0.5px solid var(--color-border-secondary)",
                background: "transparent", color: "var(--color-text-tertiary)", cursor: "pointer",
              }}
            >
              ↻ Refresh
            </button>
          </div>
        </div>
      </div>

      {/* Prediction banner */}
      <PredictionBanner
        result={predictionResult}
        onDismiss={() => setPredictionResult(null)}
      />

      {/* Tabs */}
      <div style={{ display: "flex", gap: 4, marginBottom: 16, overflowX: "auto", paddingBottom: 4 }}>
        {TABS.map((tab) => {
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                display: "flex", alignItems: "center", gap: 6, flexShrink: 0,
                padding: "7px 14px", borderRadius: 7, border: "none", cursor: "pointer",
                fontSize: 12, fontWeight: isActive ? 600 : 400, transition: "all .15s",
                background: isActive ? "var(--color-background-primary)" : "transparent",
                color: isActive ? "var(--color-text-primary)" : "var(--color-text-secondary)",
                boxShadow: isActive ? "0 0 0 0.5px var(--color-border-secondary)" : "none",
              }}
            >
              <span style={{ fontSize: 13 }}>{tab.icon}</span>
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Panels */}
      <div>
        {activeTab === 0 && (
          <OverviewPanel
            ckVals={ckVals} setCkVals={setCkVals}
            fgVals={fgVals} setFgVals={setFgVals}
            dirVals={dirVals} setDirVals={setDirVals}
            rsVals={rsVals} setRsVals={setRsVals}
            rhVals={rhVals}
            tcVals={tcVals} setTcVals={setTcVals}
            tmVals={tmVals} setTmVals={setTmVals}
            eventId={eventId} onGoTab={setActiveTab}
            onPredict={handlePredictionResult}
          />
        )}
        {activeTab === 1 && <CentroidKinematicsPanel vals={ckVals} setVals={setCkVals} />}
        {activeTab === 2 && <FireGrowthPanel vals={fgVals} setVals={setFgVals} />}
        {activeTab === 3 && <DirectionalityPanel vals={dirVals} setVals={setDirVals} />}
        {activeTab === 4 && <ProximityPanel vals={rsVals} setVals={setRsVals} />}
        {activeTab === 5 && rhVals && <RiskHorizonPanel vals={rhVals} rsVals={rsVals} />}
      </div>

      {showUpload && (
        <JsonUploadModal
          onApply={handleUploadApply}
          onClose={() => setShowUpload(false)}
        />
      )}
    </div>
  );
}