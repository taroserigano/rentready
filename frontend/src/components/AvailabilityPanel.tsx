import { Users } from "lucide-react";
import type { Slot, TourAgent } from "../types";

export function AvailabilityPanel({
  staff,
  slots,
  loading,
}: {
  staff: TourAgent[];
  slots: Slot[];
  loading: boolean;
}) {
  // Open-slot counts per agent, so each staffer's meter is relative to
  // whoever has the most availability this window.
  const counts = new Map<string, number>();
  for (const s of slots) counts.set(s.agent_id, (counts.get(s.agent_id) ?? 0) + 1);
  const max = Math.max(1, ...counts.values());

  return (
    <div className="card">
      <h2 className="icon-line">
        <Users size={16} className="icon-muted" /> Team availability
      </h2>

      {loading && <p className="muted">Loading availability…</p>}
      {!loading && staff.length === 0 && (
        <p className="muted">No leasing staff cover this property yet.</p>
      )}

      {!loading &&
        staff.map((a) => {
          const open = counts.get(a.id) ?? 0;
          const pct = Math.round((open / max) * 100);
          const tone = open === 0 ? "bad" : open <= 2 ? "warn" : "good";
          return (
            <div
              className="subpanel"
              key={a.id}
              style={{ marginTop: "var(--s-2)" }}
            >
              <div className="mini-row">
                <span className="v" style={{ textAlign: "left" }}>
                  {a.name}
                </span>
                <span
                  className={`badge tone-${open > 0 ? "good" : "bad"}`}
                  aria-label={`${open} open ${open === 1 ? "slot" : "slots"}`}
                >
                  {open} open
                </span>
              </div>
              <div className="mini-row">
                <span className="k">{a.role}</span>
              </div>
              {a.areas.length > 0 && (
                <div className="mini-row">
                  <span className="k">{a.areas.join(", ")}</span>
                </div>
              )}
              <div className="meter" style={{ marginTop: 6 }}>
                <i
                  className={tone}
                  style={{ width: `${open === 0 ? 0 : Math.max(pct, 8)}%` }}
                />
              </div>
            </div>
          );
        })}
    </div>
  );
}
