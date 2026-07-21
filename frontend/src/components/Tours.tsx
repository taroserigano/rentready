import { useCallback, useEffect, useRef, useState } from "react";
import type { Property, Slot, TourAgent, TourBooking } from "../types";
import {
  cancelTour,
  getOpenSlots,
  getProperties,
  getTourStaff,
  listTours,
} from "../api";
import { toast } from "../toast";
import { PropertySelector } from "./PropertySelector";
import { SchedulerChat } from "./SchedulerChat";
import { TourList } from "./TourList";
import { AvailabilityPanel } from "./AvailabilityPanel";
import { TourCalendar } from "./TourCalendar";

/** Local "YYYY-MM-DD" for a Date (avoids UTC shift from toISOString). */
function dayKey(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function errText(e: unknown): string {
  return e instanceof TypeError
    ? "Could not reach the server. Is the backend running?"
    : e instanceof Error
      ? e.message
      : String(e);
}

export function Tours({ initialPropertyId }: { initialPropertyId?: string }) {
  const [properties, setProperties] = useState<Property[]>([]);
  const [propertyId, setPropertyId] = useState<string>(initialPropertyId ?? "");
  const [locked, setLocked] = useState<boolean>(!!initialPropertyId);

  const [staff, setStaff] = useState<TourAgent[]>([]);
  const [slots, setSlots] = useState<Slot[]>([]);
  const [sideLoading, setSideLoading] = useState(false);

  const [tours, setTours] = useState<TourBooking[]>([]);
  const [toursLoading, setToursLoading] = useState(false);
  const [toursError, setToursError] = useState("");

  // The chat registers its slot-pick function here so the week calendar
  // books through the exact same flow (name prompt, toast, refetch).
  const pickRef = useRef<(slot: Slot) => void>(() => {});
  const handlePick = useCallback((slot: Slot) => pickRef.current(slot), []);

  // Property catalog for the selector.
  useEffect(() => {
    getProperties()
      .then((res) => setProperties(res.properties))
      .catch(() => setProperties([]));
  }, []);

  // React to a new deep link coming from a property page.
  useEffect(() => {
    if (initialPropertyId) {
      setPropertyId(initialPropertyId);
      setLocked(true);
    }
  }, [initialPropertyId]);

  // Staff + open slots depend on the chosen property.
  const loadSide = useCallback(async (pid: string) => {
    if (!pid) {
      setStaff([]);
      setSlots([]);
      return;
    }
    setSideLoading(true);
    try {
      // Widen the slot window to a full 7 days so the week calendar is
      // populated (the side panels only need "what's open" either way).
      const today = new Date();
      const to = new Date(today);
      to.setDate(today.getDate() + 7);
      const [s, sl] = await Promise.all([
        getTourStaff(pid),
        getOpenSlots(pid, { from: dayKey(today), to: dayKey(to) }),
      ]);
      setStaff(s);
      setSlots(sl);
    } catch {
      setStaff([]);
      setSlots([]);
    } finally {
      setSideLoading(false);
    }
  }, []);

  // Booked tours (filtered to the property when one is selected).
  const loadTours = useCallback(async (pid: string) => {
    setToursLoading(true);
    setToursError("");
    try {
      setTours(await listTours(pid ? { property_id: pid } : {}));
    } catch (e) {
      setToursError(errText(e));
      setTours([]);
    } finally {
      setToursLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSide(propertyId);
  }, [propertyId, loadSide]);

  useEffect(() => {
    loadTours(propertyId);
  }, [propertyId, loadTours]);

  function handleSelect(id: string) {
    setPropertyId(id);
    setLocked(false);
  }

  function handleBooked(booking: TourBooking) {
    toast(
      `Tour booked with ${booking.agent_name} — ${booking.property_name}`,
      "good",
    );
    loadSide(propertyId);
    loadTours(propertyId);
  }

  async function handleCancel(bookingId: string) {
    try {
      await cancelTour(bookingId);
      toast("Tour cancelled", "info");
      loadSide(propertyId);
      loadTours(propertyId);
    } catch (e) {
      toast(errText(e), "bad");
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Tour Scheduler</h1>
        <p>
          Chat to find an open time, then book a tour with a leasing agent —
          all in plain language.
        </p>
      </header>

      <div className="card">
        <PropertySelector
          properties={properties}
          value={propertyId}
          onChange={handleSelect}
          locked={locked}
          onUnlock={() => setLocked(false)}
        />
      </div>

      <div className="card">
        <h2>This week</h2>
        <TourCalendar
          slots={slots}
          bookings={tours}
          onPick={handlePick}
        />
      </div>

      <div className="tours-grid">
        <SchedulerChat
          propertyId={propertyId || undefined}
          fallbackSlots={slots}
          onBooked={handleBooked}
          registerPick={(pick) => {
            pickRef.current = pick;
          }}
        />
        <div className="right-col">
          <TourList
            tours={tours}
            loading={toursLoading}
            error={toursError}
            onCancel={handleCancel}
          />
          <AvailabilityPanel staff={staff} slots={slots} loading={sideLoading} />
        </div>
      </div>
    </div>
  );
}
