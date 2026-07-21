import { Lock } from "lucide-react";
import type { Property } from "../types";

export function PropertySelector({
  properties,
  value,
  onChange,
  locked,
  onUnlock,
}: {
  properties: Property[];
  value: string;
  onChange: (id: string) => void;
  /** True when the page was deep-linked from a property page. */
  locked: boolean;
  onUnlock: () => void;
}) {
  const selected = properties.find((p) => p.id === value);

  if (locked && selected) {
    return (
      <div className="field">
        <span>Property</span>
        <div className="mini-row" style={{ padding: 0 }}>
          <span className="badge tone-info icon-line">
            <Lock size={12} /> {selected.name}
          </span>
          <button className="btn-ghost btn-small" onClick={onUnlock}>
            Change
          </button>
        </div>
      </div>
    );
  }

  return (
    <label className="field">
      <span>Property</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="">Select a property…</option>
        {properties.map((p) => (
          <option key={p.id} value={p.id}>
            {p.name} · {p.area}
          </option>
        ))}
      </select>
    </label>
  );
}
