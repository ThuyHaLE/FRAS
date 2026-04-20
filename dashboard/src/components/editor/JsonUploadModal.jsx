// dashboard/src/components/editor/JsonUploadModal.jsx
// A modal dialog for uploading or pasting JSON data, which will be sent to the API for prediction. 

import React, { useState } from "react";
import { ACCENT } from "../../constants/featureRanges";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export function JsonUploadModal({ onApply, onClose }) {
  const [text, setText]       = useState("");
  const [error, setError]     = useState(null);
  const [loading, setLoading] = useState(false);
  const fileRef = React.useRef();

  function handleFile(e) {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => { setText(ev.target.result); setError(null); };
    reader.readAsText(file);
  }

  async function handleApply() {
    // 1. Validate JSON
    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch {
      setError("Invalid JSON. Please check your input.");
      return;
    }

    // 2. Call API predict
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ event: parsed }),
      });
      const data = await res.json();
      if (!res.ok || !data.ok) {
        throw new Error(data.detail ?? `HTTP ${res.status}`);
      }
      onApply(data.result);   // transfer prediction to dashboard
      onClose();
    } catch (e) {
      setError(`Predict error: ${e.message}`);
    } finally {
      setLoading(false);
    }
  }

  const canApply = text.trim() && !loading;

  return (
    <div style={{
      position: "fixed", inset: 0, zIndex: 100,
      background: "rgba(0,0,0,0.45)",
      display: "flex", alignItems: "center", justifyContent: "center",
    }}>
      <div style={{
        background: "var(--color-background-primary)",
        border: "0.5px solid var(--color-border-secondary)",
        borderRadius: 12, padding: "1.25rem 1.5rem",
        width: "min(560px, 92vw)", boxSizing: "border-box",
      }}>
        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
          <div style={{ fontSize: 13, fontWeight: 500, color: "var(--color-text-primary)" }}>
            Upload / paste JSON — Predict
          </div>
          <button onClick={onClose} style={{ background: "none", border: "none", cursor: "pointer", fontSize: 16, color: "var(--color-text-tertiary)", lineHeight: 1 }}>✕</button>
        </div>

        {/* File picker */}
        <div
          onClick={() => fileRef.current.click()}
          style={{
            border: "0.5px dashed var(--color-border-secondary)", borderRadius: 8,
            padding: "12px 16px", marginBottom: 10, cursor: "pointer",
            background: "var(--color-background-secondary)",
            display: "flex", alignItems: "center", gap: 10,
          }}
        >
          <span style={{ fontSize: 18 }}>📂</span>
          <div>
            <div style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-primary)" }}>Choose .json file</div>
            <div style={{ fontSize: 10, color: "var(--color-text-tertiary)" }}>or drag and drop here</div>
          </div>
          <input ref={fileRef} type="file" accept=".json,application/json" onChange={handleFile} style={{ display: "none" }} />
        </div>

        <div style={{ fontSize: 10, color: "var(--color-text-tertiary)", textAlign: "center", marginBottom: 8 }}>
          — or paste JSON directly —
        </div>

        {/* Textarea */}
        <textarea
          value={text}
          onChange={(e) => { setText(e.target.value); setError(null); }}
          placeholder={'{\n  "fire_area_ha": 42,\n  "wind_speed_kmh": 35,\n  ...\n}'}
          style={{
            width: "100%", height: 180, boxSizing: "border-box",
            fontSize: 11, fontFamily: "var(--font-mono)",
            padding: "8px 10px", borderRadius: 6,
            border: `0.5px solid ${error ? "#c44" : "var(--color-border-secondary)"}`,
            background: "var(--color-background-secondary)",
            color: "var(--color-text-primary)", resize: "vertical",
          }}
        />

        {/* Error */}
        {error && (
          <div style={{ fontSize: 11, color: "#c44", marginTop: 4 }}>{error}</div>
        )}

        {/* Loading indicator */}
        {loading && (
          <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginTop: 6, display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ animation: "spin 1s linear infinite", display: "inline-block" }}>⏳</span>
            Đang gọi model predict…
            <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
          </div>
        )}

        {/* Actions */}
        <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 12 }}>
          <button
            onClick={onClose}
            disabled={loading}
            style={{
              fontSize: 12, padding: "6px 14px", borderRadius: 6,
              border: "0.5px solid var(--color-border-secondary)",
              background: "transparent", color: "var(--color-text-secondary)",
              cursor: loading ? "default" : "pointer", opacity: loading ? 0.5 : 1,
            }}
          >
            Hủy
          </button>
          <button
            onClick={handleApply}
            disabled={!canApply}
            style={{
              fontSize: 12, padding: "6px 14px", borderRadius: 6, border: "none",
              background: canApply ? ACCENT : "var(--color-background-secondary)",
              color: canApply ? "#fff" : "var(--color-text-tertiary)",
              cursor: canApply ? "pointer" : "default", fontWeight: 500,
              minWidth: 64,
            }}
          >
            {loading ? "…" : "Apply"}
          </button>
        </div>
      </div>
    </div>
  );
}