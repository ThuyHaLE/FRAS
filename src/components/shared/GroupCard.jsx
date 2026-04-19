// src/components/shared/GroupCard.jsx
// A card that groups features together, with a header and a list of FeatureCards.

import { FeatureCard } from "./FeatureCard";

export function GroupCard({ group, features, vals, activeFeature, onToggle, onUpdate, extra }) {
  return (
    <div style={{ background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: 10, padding: ".9rem 1rem" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 12, paddingBottom: 8, borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
        <div style={{ width: 9, height: 9, borderRadius: 2, flexShrink: 0, background: group.dotColor }} />
        <div style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-primary)" }}>{group.label}</div>
      </div>
      {features.map((f) => (
        <FeatureCard key={f.id} feature={f} value={vals[f.id]} isActive={activeFeature === f.id} onToggle={onToggle} onUpdate={onUpdate} />
      ))}
      {extra}
    </div>
  );
}