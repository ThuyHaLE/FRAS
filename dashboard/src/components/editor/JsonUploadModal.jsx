// src/components/editor/JsonUploadModal.jsx
// Model alowing users to upload or paste JSON parameters for the model, with basic validation and error handling.
// This component is used in the Editor page to allow users to quickly load parameters from a JSON file or text input, 
// which can be useful for sharing configurations or restoring previous states. 
// It provides a simple interface with a file input and a textarea, along with error feedback if the JSON is invalid. 

import React, { useState } from "react";
import { ACCENT } from "./constants/featureRanges"; 

export function JsonUploadModal({ onApply, onClose }) {
  const [text, setText] = useState("");
  const [error, setError] = useState(null);
  const fileRef = React.useRef();
  function handleFile(e) {
    const file = e.target.files[0]; if (!file) return;
    const reader = new FileReader(); reader.onload = (ev) => setText(ev.target.result); reader.readAsText(file);
  }
  function handleApply() { try { onApply(JSON.parse(text)); onClose(); } catch { setError("JSON không hợp lệ — kiểm tra lại cú pháp."); } }
  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 100, background: "rgba(0,0,0,0.45)", display: "flex", alignItems: "center", justifyContent: "center" }}>
      <div style={{ background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-secondary)", borderRadius: 12, padding: "1.25rem 1.5rem", width: "min(560px, 92vw)", boxSizing: "border-box" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
          <div style={{ fontSize: 13, fontWeight: 500, color: "var(--color-text-primary)" }}>Upload / paste JSON parameters</div>
          <button onClick={onClose} style={{ background: "none", border: "none", cursor: "pointer", fontSize: 16, color: "var(--color-text-tertiary)", lineHeight: 1 }}>✕</button>
        </div>
        <div onClick={() => fileRef.current.click()} style={{ border: "0.5px dashed var(--color-border-secondary)", borderRadius: 8, padding: "12px 16px", marginBottom: 10, cursor: "pointer", background: "var(--color-background-secondary)", display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 18 }}>📂</span>
          <div>
            <div style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-primary)" }}>Chọn file .json</div>
            <div style={{ fontSize: 10, color: "var(--color-text-tertiary)" }}>hoặc kéo thả vào đây</div>
          </div>
          <input ref={fileRef} type="file" accept=".json,application/json" onChange={handleFile} style={{ display: "none" }} />
        </div>
        <div style={{ fontSize: 10, color: "var(--color-text-tertiary)", textAlign: "center", marginBottom: 8 }}>— hoặc paste JSON trực tiếp —</div>
        <textarea value={text} onChange={(e) => { setText(e.target.value); setError(null); }}
          placeholder={'{\n  "fireGrowth": { "area_first_ha": 42, ... },\n  ...\n}'}
          style={{ width: "100%", height: 180, boxSizing: "border-box", fontSize: 11, fontFamily: "var(--font-mono)", padding: "8px 10px", borderRadius: 6, border: `0.5px solid ${error ? "#c44" : "var(--color-border-secondary)"}`, background: "var(--color-background-secondary)", color: "var(--color-text-primary)", resize: "vertical" }}
        />
        {error && <div style={{ fontSize: 11, color: "#c44", marginTop: 4 }}>{error}</div>}
        <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 12 }}>
          <button onClick={onClose} style={{ fontSize: 12, padding: "6px 14px", borderRadius: 6, border: "0.5px solid var(--color-border-secondary)", background: "transparent", color: "var(--color-text-secondary)", cursor: "pointer" }}>Hủy</button>
          <button onClick={handleApply} disabled={!text.trim()} style={{ fontSize: 12, padding: "6px 14px", borderRadius: 6, border: "none", background: text.trim() ? ACCENT : "var(--color-background-secondary)", color: text.trim() ? "#fff" : "var(--color-text-tertiary)", cursor: text.trim() ? "pointer" : "default", fontWeight: 500 }}>Apply</button>
        </div>
      </div>
    </div>
  );
}