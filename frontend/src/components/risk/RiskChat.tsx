import { useEffect, useMemo, useRef, useState } from "react";
import { Building2, Sparkles, User } from "lucide-react";
import type { RiskBand, RiskChatRequest } from "../../types";
import { askRiskChat } from "../../api";
import { BAND_LABEL, BAND_TONE } from "./riskTone";
import { RiskChatMessage, type RiskChatMsg } from "./RiskChatMessage";
import { riskStarters } from "./riskChatStarters";

type Scope = "applicant" | "portfolio";

/** A leading TypeError from fetch means the backend is unreachable. */
function errText(e: unknown): string {
  return e instanceof TypeError
    ? "Could not reach the server. Is the backend running?"
    : e instanceof Error
      ? e.message
      : String(e);
}

/**
 * Sticky right-rail chat for the Risk page. Applicant-context-aware: seeded to
 * the selected applicant, with an Applicant⇄Portfolio scope toggle. The thread
 * persists across row reselection — a "Now asking about {name}" system line is
 * injected and the starter chips re-seed. Grounded, decision-support only;
 * degrades to deterministic rules (offline badge) when no LLM is available.
 *
 * Phase 1 uses the NON-streaming `askRiskChat`; SSE streaming lands in Phase 3.
 */
export function RiskChat({
  applicantId,
  applicantName,
  band,
  onSelectApplicant,
}: {
  applicantId: string | null;
  applicantName?: string;
  band?: RiskBand;
  onSelectApplicant?: (id: string) => void;
}) {
  const [messages, setMessages] = useState<RiskChatMsg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [scope, setScope] = useState<Scope>(
    applicantId ? "applicant" : "portfolio",
  );
  const idRef = useRef(0);
  // Track the applicant the thread is currently seeded to, to detect changes.
  const prevApplicant = useRef(applicantId);

  const who = applicantName?.trim() || "this applicant";
  const effectiveScope: Scope = applicantId ? scope : "portfolio";

  // React to the selected applicant changing: keep the thread, inject a system
  // line, and re-point the scope at the new applicant (re-seeds the starters).
  useEffect(() => {
    if (prevApplicant.current === applicantId) return;
    prevApplicant.current = applicantId;
    setScope(applicantId ? "applicant" : "portfolio");
    const line = applicantId
      ? `Now asking about ${applicantName?.trim() || "this applicant"}.`
      : "Now viewing the portfolio.";
    const sysId = ++idRef.current;
    setMessages((m) =>
      // Skip the line on the very first seed (empty thread).
      m.length === 0 ? m : [...m, { id: sysId, who: "system", text: line }],
    );
  }, [applicantId, applicantName]);

  const starters = useMemo(
    () => riskStarters(applicantName, band, effectiveScope),
    [applicantName, band, effectiveScope],
  );

  // Starters show on a fresh thread and again right after a context switch.
  const showStarters =
    messages.length === 0 ||
    messages[messages.length - 1]?.who === "system";

  /** Update a single message in place by id. */
  function patchMsg(id: number, fn: (m: RiskChatMsg) => RiskChatMsg) {
    setMessages((ms) => ms.map((m) => (m.id === id ? fn(m) : m)));
  }

  async function send(q: string) {
    q = q.trim();
    if (!q || busy) return;
    // History is the conversation so far (before this turn), oldest first.
    // System context lines are UI-only and never sent to the model.
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

    const request: RiskChatRequest = {
      question: q,
      applicant_id:
        effectiveScope === "applicant" && applicantId
          ? applicantId
          : undefined,
      history,
    };

    try {
      const res = await askRiskChat(request);
      patchMsg(botId, (m) => ({
        ...m,
        text: res.answer,
        pending: false,
        res,
      }));
    } catch (e) {
      patchMsg(botId, (m) => ({
        ...m,
        text: `Sorry — ${errText(e)}`,
        pending: false,
      }));
    } finally {
      setBusy(false);
    }
  }

  const scopeTone = band ? BAND_TONE[band] : "info";

  return (
    <div className="card" style={{ marginTop: 0 }}>
      <div className="rec-head">
        <h2 style={{ margin: 0 }}>Ask about risk</h2>
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
        {effectiveScope === "applicant" ? (
          <span className={`badge tone-${scopeTone} icon-line`}>
            <User size={12} /> Ask about {who}
            {band ? ` · ${BAND_LABEL[band]}` : ""}
          </span>
        ) : (
          <span className="badge tone-info icon-line">
            <Building2 size={12} /> Portfolio view
          </span>
        )}

        {/* Applicant⇄Portfolio scope toggle (Applicant disabled with no selection). */}
        <div className="chip-row" style={{ marginTop: 0, marginLeft: "auto" }}>
          <button
            type="button"
            className="chip"
            aria-pressed={effectiveScope === "applicant"}
            disabled={!applicantId}
            style={
              effectiveScope === "applicant"
                ? {
                    borderColor: "var(--accent)",
                    color: "var(--accent-text)",
                    background: "var(--accent-soft)",
                  }
                : undefined
            }
            onClick={() => setScope("applicant")}
          >
            Applicant
          </button>
          <button
            type="button"
            className="chip"
            aria-pressed={effectiveScope === "portfolio"}
            style={
              effectiveScope === "portfolio"
                ? {
                    borderColor: "var(--accent)",
                    color: "var(--accent-text)",
                    background: "var(--accent-soft)",
                  }
                : undefined
            }
            onClick={() => setScope("portfolio")}
          >
            Portfolio
          </button>
        </div>
      </div>

      <div aria-live="polite" style={{ marginTop: 12 }}>
        {messages.length === 0 && (
          <p className="muted" style={{ marginTop: 0 }}>
            Ask why an applicant is scored the way they are, what would lower the
            risk, or how the model works. Answers are grounded and cite the
            model's own reason codes.
          </p>
        )}
        {messages.map((m) =>
          m.who === "bot" ? (
            <RiskChatMessage
              key={m.id}
              msg={m}
              onFollowUp={send}
              onSelectApplicant={onSelectApplicant}
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
            <button
              key={s}
              className="chip"
              onClick={() => send(s)}
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
          send(input);
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={
            effectiveScope === "applicant"
              ? `e.g. Why is ${who} scored this way?`
              : "e.g. Which factors are excluded from the model?"
          }
        />
        <button type="submit" disabled={busy}>
          Ask
        </button>
      </form>
    </div>
  );
}
