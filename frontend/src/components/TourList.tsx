import { useState } from "react";
import { CalendarDays, MapPin, User } from "lucide-react";
import type { TourBooking } from "../types";
import { toursIcsUrl } from "../api";

/** "Tue Jul 22, 2:00 PM" from an ISO-local start string. */
function whenLabel(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

export function TourList({
  tours,
  loading,
  error,
  onCancel,
}: {
  tours: TourBooking[];
  loading: boolean;
  error: string;
  onCancel: (bookingId: string) => Promise<void> | void;
}) {
  const [cancelling, setCancelling] = useState<string | null>(null);

  async function handleCancel(id: string) {
    setCancelling(id);
    try {
      await onCancel(id);
    } finally {
      setCancelling(null);
    }
  }

  return (
    <div className="card">
      <h2 className="icon-line">
        <CalendarDays size={16} className="icon-muted" /> Upcoming tours
      </h2>

      {loading && <p className="muted">Loading tours…</p>}
      {!loading && error && (
        <div className="error" role="alert" style={{ marginTop: 0 }}>
          {error}
        </div>
      )}
      {!loading && !error && tours.length === 0 && (
        <p className="muted">No tours booked yet. Pick a time to schedule one.</p>
      )}

      {!loading &&
        !error &&
        tours.map((t) => (
          <div className="tour-item" key={t.id}>
            <div className="tour-when">
              <CalendarDays size={13} /> {whenLabel(t.start)}
            </div>
            <div className="tour-sub icon-line">
              <MapPin size={12} /> {t.property_name}
            </div>
            <div className="tour-sub icon-line">
              <User size={12} /> {t.agent_name}
              {t.prospect_name ? ` · ${t.prospect_name}` : ""}
            </div>
            <div className="cal-links">
              {t.gcal_url && (
                <a
                  className="btn btn-ghost btn-small"
                  href={t.gcal_url}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Add to Google Calendar
                </a>
              )}
              <a className="btn btn-ghost btn-small" href={toursIcsUrl(t.id)}>
                .ics
              </a>
            </div>
            <div className="tour-actions">
              <button
                className="btn-ghost btn-small danger"
                disabled={cancelling === t.id}
                onClick={() => handleCancel(t.id)}
              >
                {cancelling === t.id ? "Cancelling…" : "Cancel"}
              </button>
            </div>
          </div>
        ))}
    </div>
  );
}
