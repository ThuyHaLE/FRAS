// src/components/shared/MiniStat.jsx
// A small card component for displaying a single statistic with a label and optional subtext.

export function MiniStat({ label, value, sub }) {
  return (
    <div style={{ background: "var(--color-background-secondary)", borderRadius: 8, padding: ".85rem 1rem" }}>
      <div style={{ fontSize: 11, color: "var(--color-text-secondary)", marginBottom: 3 }}>{label}</div>
      <div style={{ fontSize: 20, fontWeight: 500, color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginTop: 2, fontFamily: "var(--font-mono)" }}>{sub}</div>}
    </div>
  );
}