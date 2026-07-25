import { useEffect, useRef, useState } from "react";
import { Database, FileText, Sparkles } from "lucide-react";
import type { AskResponse } from "../types";
import { Feedback } from "./Feedback";
import { Markdown } from "./Markdown";

interface Msg {
  /** Stable id (not derived from text) so feedback can target this exact
   * message even when two answers share a common prefix. */
  id: number;
  who: "user" | "bot";
  text: string;
  res?: AskResponse;
}

const SUGGESTIONS = [
  "What is the applicant's monthly income?",
  "Any eviction or bankruptcy history?",
  "Can they afford $1,800/mo?",
  "What areas do they prefer?",
];

/** Map the backend `source` string to a human label + icon. */
function sourceMeta(source: string): { label: string; icon: JSX.Element } {
  const s = (source || "").toLowerCase();
  if (s.includes("graph"))
    return { label: "Graph", icon: <Database size={12} /> };
  if (s.includes("anthropic") || s.includes("claude"))
    return { label: "Claude + documents", icon: <FileText size={12} /> };
  if (s.includes("mock") || s.includes("retrieve"))
    return { label: "Offline (retrieve-only)", icon: <FileText size={12} /> };
  return { label: source || "source", icon: <FileText size={12} /> };
}

function BotAnswer({
  msg,
  applicantId,
}: {
  msg: Msg;
  applicantId?: string;
}) {
  const [showCites, setShowCites] = useState(false);
  const res = msg.res;
  const meta = res ? sourceMeta(res.source) : null;
  const sources = res?.sources ?? [];
  return (
    <div className="chat-msg bot">
      <Markdown text={msg.text} />
      {res && (
        <>
          <div className="chat-actions">
            {meta && (
              <span className="badge tone-info icon-line" title={`Answered from: ${meta.label}`}>
                {meta.icon}
                {meta.label}
              </span>
            )}
            {sources.length > 0 && (
              <button
                className="cite-toggle"
                aria-expanded={showCites}
                onClick={() => setShowCites((v) => !v)}
              >
                {showCites ? "Hide" : "Show"} {sources.length} source
                {sources.length > 1 ? "s" : ""}
              </button>
            )}
            <span style={{ marginLeft: "auto" }}>
              <Feedback applicantId={applicantId} target="ask" itemId={String(msg.id)} />
            </span>
          </div>
          {showCites && sources.length > 0 && (
            <div className="cite-list">
              {sources.map((s, i) => (
                <div className="cite-item" key={i}>
                  <span className="cite-num">{i + 1}</span>
                  <span>{s}</span>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

export function Chat({
  onAsk,
  applicantId,
}: {
  onAsk: (q: string) => Promise<AskResponse>;
  applicantId?: string;
}) {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const idRef = useRef(0);
  // Guards against updating state after the applicant-detail view (or this
  // chat) unmounts while `onAsk` is still in flight.
  const aliveRef = useRef(true);
  useEffect(() => {
    aliveRef.current = true;
    return () => {
      aliveRef.current = false;
    };
  }, []);

  async function send(q: string) {
    q = q.trim();
    if (!q || busy) return;
    setMessages((m) => [...m, { id: ++idRef.current, who: "user", text: q }]);
    setInput("");
    setBusy(true);
    try {
      const res = await onAsk(q);
      if (!aliveRef.current) return;
      setMessages((m) => [...m, { id: ++idRef.current, who: "bot", text: res.answer, res }]);
    } catch (err) {
      if (!aliveRef.current) return;
      setMessages((m) => [...m, { id: ++idRef.current, who: "bot", text: `Error: ${err}` }]);
    } finally {
      if (aliveRef.current) setBusy(false);
    }
  }

  return (
    <div className="card">
      <h2>5. Ask about this application</h2>
      <div aria-live="polite">
        {messages.map((m) =>
          m.who === "bot" ? (
            <BotAnswer key={m.id} msg={m} applicantId={applicantId} />
          ) : (
            <div key={m.id} className="chat-msg user">
              {m.text}
            </div>
          ),
        )}
        {busy && (
          <div className="chat-msg bot" aria-label="Assistant is thinking">
            <span className="chat-typing">
              <span />
              <span />
              <span />
            </span>
          </div>
        )}
      </div>

      {messages.length === 0 && (
        <div className="chip-row">
          {SUGGESTIONS.map((s) => (
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
          placeholder="e.g. What is the applicant's income?"
        />
        <button type="submit" disabled={busy}>
          Ask
        </button>
      </form>
    </div>
  );
}
