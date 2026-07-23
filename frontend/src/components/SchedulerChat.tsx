import { useEffect, useRef, useState } from "react";
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

/** These three phases collect contact details one field at a time server-side;
 * the frontend instead shows them all as one form (see CONTACT_PHASES below). */
type ContactPhase = "awaiting_name" | "awaiting_phone" | "awaiting_email";
const CONTACT_PHASES: ContactPhase[] = ["awaiting_name", "awaiting_phone", "awaiting_email"];

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
  const [contact, setContact] = useState({ name: "", phone: "", email: "" });

  const ready = !!propertyId;
  const contactPhase = CONTACT_PHASES.includes(state?.phase as ContactPhase)
    ? (state!.phase as ContactPhase)
    : null;

  // Seed the contact form once when this stage begins (e.g. name may already
  // be captured if a phone/email retry brought us back through this phase).
  const seededRef = useRef(false);
  useEffect(() => {
    if (contactPhase && !seededRef.current) {
      setContact({
        name: state?.prospect_name ?? "",
        phone: state?.prospect_phone ?? "",
        email: state?.prospect_email ?? "",
      });
      seededRef.current = true;
    }
    if (!contactPhase) seededRef.current = false;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [contactPhase]);

  /** One turn: append the user's text, call the backend, apply the reply.
   * Threads messages/state explicitly (rather than reading component state)
   * so callers can chain several turns in a row without a stale-closure race. */
  async function sendTurn(
    priorMessages: ChatMessage[],
    priorState: ChatState | undefined,
    userText: string,
    selectedSlotId?: string,
  ): Promise<{ messages: ChatMessage[]; state?: ChatState; booking?: TourBooking | null }> {
    const next = [...priorMessages, { role: "user", content: userText } as ChatMessage];
    setMessages(next);
    try {
      const res = await tourChat({
        messages: next,
        property_id: propertyId,
        state: priorState,
        selected_slot_id: selectedSlotId,
      });
      const withReply = [...next, { role: "assistant", content: res.reply } as ChatMessage];
      setMessages(withReply);
      setState(res.state);
      if (res.booking) {
        onBooked(res.booking);
        setSlots([]);
        setSelectedId("");
      } else {
        setSlots(res.proposed_slots ?? []);
      }
      return { messages: withReply, state: res.state, booking: res.booking };
    } catch {
      // The live path degrades server-side and never throws; this is a
      // defensive net for a true network outage. Keep booking discoverable
      // by surfacing the slots the parent already loaded.
      const withReply = [...next, { role: "assistant", content: CANNED_REPLY } as ChatMessage];
      setMessages(withReply);
      setSlots(fallbackSlots);
      setSelectedId("");
      return { messages: withReply, state: priorState, booking: null };
    }
  }

  async function runTurn(userText: string, selectedSlotId?: string) {
    const text = userText.trim();
    if ((!text && !selectedSlotId) || busy || !ready) return;
    setInput("");
    setBusy(true);
    setSelectedId(selectedSlotId ?? "");
    try {
      await sendTurn(messages, state, text, selectedSlotId);
    } finally {
      setBusy(false);
    }
  }

  function pickSlot(slot: Slot) {
    runTurn(slot.label, slot.slot_id);
  }

  /** Submit the name/phone/email form as up to three chained turns — only
   * the field(s) the backend still needs (by CURRENT phase after each turn),
   * so a corrected retry never resends an already-accepted field. */
  async function submitContact(e: React.FormEvent) {
    e.preventDefault();
    if (busy || !ready) return;
    setBusy(true);
    try {
      let m = messages;
      let s = state;
      const fields: Array<[ContactPhase, string]> = [
        ["awaiting_name", contact.name.trim()],
        ["awaiting_phone", contact.phone.trim()],
        ["awaiting_email", contact.email.trim()],
      ];
      for (const [phase, value] of fields) {
        if (!value || s?.phase !== phase) continue;
        const r = await sendTurn(m, s, value);
        m = r.messages;
        s = r.state;
        if (r.booking) break;
      }
    } finally {
      setBusy(false);
    }
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

      {contactPhase ? (
        <form className="form-grid" onSubmit={submitContact} style={{ marginTop: 12 }}>
          <label className="field">
            <span>Name</span>
            <input
              value={contact.name}
              onChange={(e) => setContact((c) => ({ ...c, name: e.target.value }))}
              placeholder="Jane Doe"
              autoComplete="name"
              disabled={busy}
              required
            />
          </label>
          <label className="field">
            <span>Phone</span>
            <input
              type="tel"
              value={contact.phone}
              onChange={(e) => setContact((c) => ({ ...c, phone: e.target.value }))}
              placeholder="(512) 555-0100"
              autoComplete="tel"
              disabled={busy}
              required
            />
          </label>
          <label className="field">
            <span>Email</span>
            <input
              type="email"
              value={contact.email}
              onChange={(e) => setContact((c) => ({ ...c, email: e.target.value }))}
              placeholder="jane@example.com"
              autoComplete="email"
              disabled={busy}
              required
            />
          </label>
          <div style={{ gridColumn: "1 / -1" }}>
            <button type="submit" disabled={busy}>
              {busy ? "Booking…" : "Confirm tour"}
            </button>
          </div>
        </form>
      ) : (
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
      )}
    </div>
  );
}
