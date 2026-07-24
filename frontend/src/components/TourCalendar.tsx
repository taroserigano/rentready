import { useMemo } from "react";
import type { Slot, TourBooking } from "../types";

interface TourCalendarProps {
  /** Open slots for the week (from parent). */
  slots: Slot[];
  /** Booked tours (from parent). */
  bookings: TourBooking[];
  /** Same handler SlotButtons uses (books via chat). */
  onPick: (slot: Slot) => void;
  selectedSlotId?: string;
  /** How many day columns to show, starting today. */
  days?: number;
}

/** Local "YYYY-MM-DD" for a Date (avoids UTC shift from toISOString). */
function dayKey(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

/** "TUE · JUL 21" header for a day key. */
function headLabel(key: string): string {
  const d = fromKey(key);
  const wd = d.toLocaleDateString(undefined, { weekday: "short" });
  const mo = d.toLocaleDateString(undefined, { month: "short" });
  return `${wd.toUpperCase()} · ${mo.toUpperCase()} ${d.getDate()}`;
}

/** Parse an ISO-local date key into a local Date (midnight). */
function fromKey(key: string): Date {
  const [y, m, d] = key.split("-").map(Number);
  return new Date(y, (m ?? 1) - 1, d ?? 1);
}

/** "2:00 PM" from an ISO-local start string. */
function timeLabel(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
}

/** Full "Tue, Jul 21, 2:00 PM" for aria labels. */
function fullLabel(iso: string): string {
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

type Item =
  | { kind: "slot"; start: string; slot: Slot }
  | { kind: "booking"; start: string; booking: TourBooking };

export function TourCalendar({
  slots,
  bookings,
  onPick,
  selectedSlotId,
  days = 7,
}: TourCalendarProps) {
  // Cheap on every render, and gives the memo below an explicit dependency
  // for "today" — recomputing once when the calendar date actually rolls
  // over, instead of freezing on whatever day the component first mounted.
  const todayKey = dayKey(new Date());

  // Bucket slots + bookings by their calendar day. Memoized so an unrelated
  // re-render of the Tours page (e.g. a toast elsewhere) doesn't redo this
  // date math — only slots/bookings/days/todayKey changing does.
  const { keys, byDay } = useMemo(() => {
    // Build the ordered list of day keys: today .. today + days - 1.
    const today = fromKey(todayKey);
    const keys: string[] = [];
    for (let i = 0; i < days; i++) {
      const d = new Date(today);
      d.setDate(today.getDate() + i);
      keys.push(dayKey(d));
    }

    const byDay = new Map<string, Item[]>();
    for (const k of keys) byDay.set(k, []);
    for (const s of slots) {
      const k = s.start.slice(0, 10);
      if (byDay.has(k)) byDay.get(k)!.push({ kind: "slot", start: s.start, slot: s });
    }
    for (const b of bookings) {
      if (b.status !== "booked") continue;
      const k = b.start.slice(0, 10);
      if (byDay.has(k))
        byDay.get(k)!.push({ kind: "booking", start: b.start, booking: b });
    }
    for (const k of keys) {
      byDay.get(k)!.sort((a, z) => a.start.localeCompare(z.start));
    }
    return { keys, byDay };
  }, [slots, bookings, days, todayKey]);

  const hasAny = slots.length > 0 || bookings.length > 0;
  if (!hasAny) {
    return (
      <p className="muted">
        Pick a property above to see the week&rsquo;s open times and booked tours.
      </p>
    );
  }

  return (
    <div className="cal-week" role="group" aria-label="Tour week calendar">
      {keys.map((k) => {
        const items = byDay.get(k)!;
        return (
          <div className="cal-day" key={k}>
            <div className="cal-day-head eyebrow">{headLabel(k)}</div>
            {items.length === 0 && <div className="cal-empty">—</div>}
            {items.map((it) => {
              if (it.kind === "slot") {
                const s = it.slot;
                const selected = s.slot_id === selectedSlotId;
                return (
                  <button
                    key={s.slot_id}
                    type="button"
                    className={`slot cal-slot${selected ? " selected" : ""}`}
                    aria-pressed={selected}
                    aria-label={`Book ${fullLabel(s.start)} with ${s.agent_name}`}
                    onClick={() => onPick(s)}
                  >
                    <span className="slot-time">{timeLabel(s.start)}</span>
                    <span className="slot-agent">{s.agent_name}</span>
                  </button>
                );
              }
              const b = it.booking;
              return (
                <div
                  className="cal-booked"
                  key={b.id}
                  aria-label={`Booked: ${fullLabel(b.start)}, ${b.property_name} with ${b.agent_name}`}
                >
                  <span className="cal-booked-time">{timeLabel(b.start)}</span>
                  <span className="cal-booked-sub">{b.property_name}</span>
                  <span className="cal-booked-sub">{b.agent_name}</span>
                </div>
              );
            })}
          </div>
        );
      })}
    </div>
  );
}
