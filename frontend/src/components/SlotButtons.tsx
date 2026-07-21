import type { Slot } from "../types";

/** Group slots by their calendar day, preserving chronological order. */
function groupByDay(slots: Slot[]): Array<{ day: string; slots: Slot[] }> {
  const order: string[] = [];
  const map = new Map<string, Slot[]>();
  for (const s of slots) {
    // "Tue Jul 22, 2:00 PM" -> "Tue Jul 22"; fall back to the ISO date.
    const day = s.label.includes(",")
      ? s.label.slice(0, s.label.lastIndexOf(","))
      : s.start.slice(0, 10);
    if (!map.has(day)) {
      map.set(day, []);
      order.push(day);
    }
    map.get(day)!.push(s);
  }
  return order.map((day) => ({ day, slots: map.get(day)! }));
}

/** Just the time portion of a slot label, e.g. "2:00 PM". */
function timeOf(s: Slot): string {
  const i = s.label.lastIndexOf(",");
  return i >= 0 ? s.label.slice(i + 1).trim() : s.label;
}

export function SlotButtons({
  slots,
  onPick,
  selectedId,
  disabled,
}: {
  slots: Slot[];
  onPick: (slot: Slot) => void;
  selectedId?: string;
  disabled?: boolean;
}) {
  if (slots.length === 0) return null;
  const groups = groupByDay(slots);
  return (
    <div role="group" aria-label="Available tour times">
      {groups.map((g) => (
        <div key={g.day} style={{ marginTop: "var(--s-2)" }}>
          <div className="eyebrow" style={{ marginBottom: 6 }}>
            {g.day}
          </div>
          <div className="slot-grid">
            {g.slots.map((s) => {
              const selected = s.slot_id === selectedId;
              return (
                <button
                  key={s.slot_id}
                  type="button"
                  className={`slot${selected ? " selected" : ""}`}
                  aria-pressed={selected}
                  aria-label={`Book ${s.label} with ${s.agent_name}`}
                  disabled={disabled}
                  onClick={() => onPick(s)}
                >
                  <span className="slot-time">{timeOf(s)}</span>
                  <span className="slot-agent">{s.agent_name}</span>
                </button>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
