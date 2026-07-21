import { useEffect, useState } from "react";
import { CalendarClock, Sparkles } from "lucide-react";
import type { ChatMessage, ChatState, Slot, TourBooking } from "../types";
import { tourChat } from "../api";
import { SlotButtons } from "./SlotButtons";

const STARTERS = [
  "I'd like to tour this week",
  "Any afternoon slots?",
  "How about this weekend?",
  "Tomorrow morning works",
];

/** Shown when the chat endpoint is unreachable (rate-limit / offline). */
const CANNED_REPLY =
  "I'm having trouble reaching the scheduler right now, but here are the open times I already have — pick one and I'll get it booked.";

export function SchedulerChat({
  propertyId,
  fallbackSlots,
  onBooked,
  registerPick,
}: {
  /** Selected property; the chat is disabled until one is chosen. */
  propertyId?: string;
  /** Parent's getOpenSlots data — rendered if the chat call fails. */
  fallbackSlots: Slot[];
  /** Fired when a turn returns a booking (parent toasts + refetches). */
  onBooked: (booking: TourBooking) => void;
  /**
   * Hands the parent this chat's slot-pick function so other surfaces
   * (e.g. the week calendar) book through the exact same chat flow.
   */
  registerPick?: (pick: (slot: Slot) => void) => void;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [state, setState] = useState<ChatState | undefined>(undefined);
  const [slots, setSlots] = useState<Slot[]>([]);
  const [selectedId, setSelectedId] = useState<string>("");

  const ready = !!propertyId;

  async function runTurn(userText: string, selectedSlotId?: string) {
    const text = userText.trim();
    if ((!text && !selectedSlotId) || busy || !ready) return;
    const next = [...messages, { role: "user", content: text } as ChatMessage];
    setMessages(next);
    setInput("");
    setBusy(true);
    setSelectedId(selectedSlotId ?? "");
    try {
      const res = await tourChat({
        messages: next,
        property_id: propertyId,
        state,
        selected_slot_id: selectedSlotId,
      });
      setMessages((m) => [...m, { role: "assistant", content: res.reply }]);
      setState(res.state);
      if (res.booking) {
        onBooked(res.booking);
        setSlots([]);
        setSelectedId("");
      } else {
        setSlots(res.proposed_slots ?? []);
      }
    } catch {
      // The live path degrades server-side and never throws; this is a
      // defensive net for a true network outage. Keep booking discoverable
      // by surfacing the slots the parent already loaded.
      setMessages((m) => [...m, { role: "assistant", content: CANNED_REPLY }]);
      setSlots(fallbackSlots);
      setSelectedId("");
    } finally {
      setBusy(false);
    }
  }

  function pickSlot(slot: Slot) {
    runTurn(slot.label, slot.slot_id);
  }

  // Expose the latest pickSlot closure to the parent every render so the
  // calendar's clicks run through this component's chat turn.
  useEffect(() => {
    registerPick?.(pickSlot);
  });

  return (
    <div className="card">
      <h2 className="icon-line">
        <CalendarClock size={16} className="icon-muted" /> Schedule a tour
      </h2>
      {!ready && (
        <p className="muted">Pick a property above to start scheduling.</p>
      )}

      <div aria-live="polite">
        {messages.map((m, i) => (
          <div
            key={i}
            className={`chat-msg ${m.role === "assistant" ? "bot" : "user"}`}
          >
            {m.content}
          </div>
        ))}
        {busy && (
          <div className="chat-msg bot" aria-label="Assistant is thinking">
            <span className="chat-typing">
              <span />
              <span />
              <span />
            </span>
          </div>
        )}
        {slots.length > 0 && (
          <SlotButtons
            slots={slots}
            onPick={pickSlot}
            selectedId={selectedId}
            disabled={busy || !ready}
          />
        )}
      </div>

      {ready && messages.length === 0 && (
        <div className="chip-row">
          {STARTERS.map((s) => (
            <button
              key={s}
              className="chip"
              onClick={() => runTurn(s)}
              disabled={busy}
            >
              <Sparkles size={13} />
              {s}
            </button>
          ))}
        </div>
      )}

      <form
        className="chat-form"
        onSubmit={(e) => {
          e.preventDefault();
          runTurn(input);
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={
            ready ? "e.g. Thursday afternoon" : "Select a property first…"
          }
          disabled={!ready || busy}
        />
        <button type="submit" disabled={!ready || busy}>
          Send
        </button>
      </form>
    </div>
  );
}
