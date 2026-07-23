import { useEffect, useMemo, useRef, useState } from "react";
import { Building2, Layers, Sparkles, User } from "lucide-react";
import type {
  ResidentChatAnswer,
  ResidentChatRequest,
  ResidentChatScope,
} from "../../types";
import { askResidentChat, askResidentChatStream } from "../../api";
import { ResidentsChatMessage, type ResidentsChatMsg } from "./ResidentsChatMessage";
import { residentStarters } from "./residentsChatStarters";
import { useEvent } from "../../useEvent";

/** A leading TypeError from fetch means the backend is unreachable. */
function errText(e: unknown): string {
  return e instanceof TypeError
    ? "Could not reach the server. Is the backend running?"
    : e instanceof Error
      ? e.message
      : String(e);
}

/**
 * Sticky right-rail chat for the Residents page. Scope-aware: seeded to the
 * selected resident, else the selected property, else the whole portfolio, with
 * a Resident⇄Property⇄Portfolio toggle (options gated by what's selected). The
 * thread persists across reselection — a "Now asking about …" system line is
 * injected and the starters re-seed. Grounded in the model's head signals only;
 * decision-support; degrades to deterministic rules (offline badge) with no LLM.
 *
 * Streams over SSE (`askResidentChatStream`): the router's `meta` frame arrives
 * first so grounded gauges / health lists paint before prose; on a transport
 * error it falls back to the non-streaming `askResidentChat`.
 */
export function ResidentsChat({
  residentId,
  residentName,
  propertyId,
  propertyName,
  onSelectProperty,
}: {
  residentId: string | null;
  residentName?: string;
  propertyId: string | null;
  propertyName?: string;
  onSelectProperty?: (propertyId: string) => void;
}) {
  const [messages, setMessages] = useState<ResidentsChatMsg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [scope, setScope] = useState<ResidentChatScope>(
    residentId ? "resident" : propertyId ? "property" : "portfolio",
  );
  const idRef = useRef(0);
  const prevResident = useRef(residentId);
  const prevProperty = useRef(propertyId);

  const who = residentName?.trim() || "this resident";
  const where = propertyName?.trim() || "this property";

  // The scope actually in force, given what's selected.
  const effectiveScope: ResidentChatScope =
    scope === "resident" && residentId
      ? "resident"
      : scope === "property" && propertyId
        ? "property"
        : scope === "resident" && propertyId
          ? "property"
          : "portfolio";

  // React to the selection changing: keep the thread, inject a system line, and
  // re-point the scope (re-seeds the starters).
  useEffect(() => {
    if (prevResident.current === residentId && prevProperty.current === propertyId) {
      return;
    }
    const residentChanged = prevResident.current !== residentId;
    prevResident.current = residentId;
    prevProperty.current = propertyId;
    const nextScope: ResidentChatScope = residentId
      ? "resident"
      : propertyId
        ? "property"
        : "portfolio";
    setScope(nextScope);
    const line = residentId
      ? `Now asking about ${residentName?.trim() || "this resident"}.`
      : propertyId
        ? `Now viewing ${propertyName?.trim() || "this property"}.`
        : "Now viewing the portfolio.";
    const sysId = ++idRef.current;
    setMessages((m) =>
      // Skip the line on the very first seed (empty thread), and skip a no-op.
      m.length === 0 || (!residentChanged && !residentId)
        ? m
        : [...m, { id: sysId, who: "system", text: line }],
    );
  }, [residentId, propertyId, residentName, propertyName]);

  const starters = useMemo(
    () => residentStarters(effectiveScope, residentName, propertyName),
    [effectiveScope, residentName, propertyName],
  );

  const showStarters =
    messages.length === 0 || messages[messages.length - 1]?.who === "system";

  /** Update a single message in place by id. */
  function patchMsg(id: number, fn: (m: ResidentsChatMsg) => ResidentsChatMsg) {
    setMessages((ms) => ms.map((m) => (m.id === id ? fn(m) : m)));
  }

  // useEvent: a permanently-stable identity so it can be passed to the
  // memo()'d ResidentsChatMessage rows without defeating their memoization.
  const send = useEvent(async (q: string) => {
    q = q.trim();
    if (!q || busy) return;
    const history = messages
      .filter((m) => m.who === "user" || m.who === "bot")
      .map((m) => ({
        role: m.who === "user" ? "user" : "assistant",
        content: m.text,
      }));
    const userId = ++idRef.current;
    const botId = ++idRef.current;
    setMessages((m) => [
      ...m,
      { id: userId, who: "user", text: q },
      { id: botId, who: "bot", text: "", pending: true },
    ]);
    setInput("");
    setBusy(true);

    const request: ResidentChatRequest = {
      question: q,
      resident_id: effectiveScope === "resident" && residentId ? residentId : undefined,
      property_id:
        effectiveScope === "property" && propertyId ? propertyId : undefined,
      history,
    };

    try {
      await askResidentChatStream(request, {
        onMeta: (meta) =>
          patchMsg(botId, (m) => ({
            ...m,
            res: {
              answer: m.text,
              scope: meta.scope,
              intent: meta.intent,
              resident_id: meta.resident_id,
              property_id: meta.property_id,
              follow_ups: meta.follow_ups,
              artifact: meta.artifact,
              source: "anthropic",
            },
          })),
        onToken: (text) =>
          patchMsg(botId, (m) => {
            const next = m.text + text;
            return {
              ...m,
              text: next,
              pending: false,
              res: m.res ? { ...m.res, answer: next } : m.res,
            };
          }),
        onDone: (source) =>
          patchMsg(botId, (m) => ({
            ...m,
            pending: false,
            res: m.res
              ? { ...m.res, source: source as ResidentChatAnswer["source"] }
              : m.res,
          })),
      });
    } catch {
      // Streaming failed (transport/unreachable) — fall back to non-streaming.
      try {
        const res = await askResidentChat(request);
        patchMsg(botId, (m) => ({ ...m, text: res.answer, pending: false, res }));
      } catch (e) {
        patchMsg(botId, (m) => ({
          ...m,
          text: `Sorry — ${errText(e)}`,
          pending: false,
        }));
      }
    } finally {
      setBusy(false);
    }
  });

  const scopeChip = (
    key: ResidentChatScope,
    label: string,
    enabled: boolean,
  ) => (
    <button
      type="button"
      className="chip"
      aria-pressed={effectiveScope === key}
      disabled={!enabled}
      style={
        effectiveScope === key
          ? {
              borderColor: "var(--accent)",
              color: "var(--accent-text)",
              background: "var(--accent-soft)",
            }
          : undefined
      }
      onClick={() => setScope(key)}
    >
      {label}
    </button>
  );

  return (
    <div className="card" style={{ marginTop: 0 }}>
      <div className="rec-head">
        <h2 style={{ margin: 0 }}>Ask about residents</h2>
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          flexWrap: "wrap",
          gap: 8,
          marginTop: 8,
        }}
      >
        {effectiveScope === "resident" ? (
          <span className="badge tone-info icon-line">
            <User size={12} /> Ask about {who}
          </span>
        ) : effectiveScope === "property" ? (
          <span className="badge tone-info icon-line">
            <Building2 size={12} /> {where}
          </span>
        ) : (
          <span className="badge tone-info icon-line">
            <Layers size={12} /> Portfolio view
          </span>
        )}

        <div className="chip-row" style={{ marginTop: 0, marginLeft: "auto" }}>
          {scopeChip("resident", "Resident", !!residentId)}
          {scopeChip("property", "Property", !!propertyId)}
          {scopeChip("portfolio", "Portfolio", true)}
        </div>
      </div>

      <div aria-live="polite" style={{ marginTop: 12 }}>
        {messages.length === 0 && (
          <p className="muted" style={{ marginTop: 0 }}>
            Ask how likely a resident is to pay late, how often or how severe it
            could get, whether they'll renew, or which apartments are healthiest.
            Answers are grounded in the model's own signals and cite its reason
            codes.
          </p>
        )}
        {messages.map((m) =>
          m.who === "bot" ? (
            <ResidentsChatMessage
              key={m.id}
              msg={m}
              onFollowUp={send}
              onSelectProperty={onSelectProperty}
            />
          ) : m.who === "system" ? (
            <div
              key={m.id}
              className="muted"
              style={{ textAlign: "center", fontSize: 12, margin: "8px 0" }}
            >
              {m.text}
            </div>
          ) : (
            <div key={m.id} className="chat-msg user">
              {m.text}
            </div>
          ),
        )}
      </div>

      {showStarters && (
        <div className="chip-row">
          {starters.map((s) => (
            <button key={s} className="chip" onClick={() => send(s)} disabled={busy}>
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
          send(input);
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={
            effectiveScope === "resident"
              ? `e.g. How likely is ${who} to pay late next year?`
              : effectiveScope === "property"
                ? `e.g. How healthy is ${where}?`
                : "e.g. Which apartments are healthiest?"
          }
        />
        <button type="submit" disabled={busy}>
          Ask
        </button>
      </form>
    </div>
  );
}
