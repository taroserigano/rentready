import { memo, useEffect, useRef, useState } from "react";
import {
  Building2,
  Check,
  ChevronDown,
  Copy,
  Database,
  DollarSign,
  FileText,
  GitCompare,
  Link2,
  MapPin,
  MessageSquare,
  Network,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  User,
  WifiOff,
  type LucideIcon,
} from "lucide-react";
import type { CompareItem, ConciergeAnswer, Property } from "../types";
import { conciergeAsk, conciergeAskStream, getProperties } from "../api";
import { Markdown } from "./Markdown";
import { LeaseViewer } from "./LeaseViewer";
import { Badge } from "./Badge";
import { TechBadge } from "./TechBadge";
import { useEvent } from "../useEvent";

/** What lease to show in the viewer, which tab, and where to scroll. */
interface LeaseTarget {
  propertyId: string;
  section?: string;
  tab?: "summary" | "document";
}

interface Msg {
  /** Stable id so streaming updates can target this exact message. */
  id: number;
  who: "user" | "bot";
  text: string;
  res?: ConciergeAnswer;
  /** Bot message awaiting its first streamed token (shows the typing dots). */
  pending?: boolean;
}

/** Matches the tone-identity palette already used for Workspace/Apply badges
 * and how-it-works icons (see index.css `--tone-*` tokens) so this page picks
 * up the same color language rather than inventing a new one. */
type Tone = "violet" | "teal" | "blue" | "magenta" | "warm";

interface StarterGroup {
  label: string;
  icon: LucideIcon;
  tone: Tone;
  items: string[];
}

/** Suggested questions, grouped so the empty state reads as a menu, not a wall
 * of chips. Every phrase is chosen to hit the backend router's keyword rules
 * (see concierge.py) so the route badge + grounding are always on-topic. */
const STARTER_GROUPS: StarterGroup[] = [
  {
    label: "Popular",
    icon: Sparkles,
    tone: "violet",
    items: [
      "What's the pet policy?",
      "How much is the security deposit?",
      "Does it have a gym?",
      "Which pet-friendly homes are under $2,000?",
    ],
  },
  {
    label: "Money & deposits",
    icon: DollarSign,
    tone: "teal",
    items: [
      "What's the monthly rent?",
      "Is there a late fee if rent is paid late?",
      "Do I need renter's insurance?",
      "What utilities are included?",
    ],
  },
  {
    label: "Amenities & home",
    icon: Building2,
    tone: "blue",
    items: [
      "Is there in-unit laundry?",
      "Is it furnished?",
      "Is there extra storage?",
      "Is there a pool?",
    ],
  },
  {
    label: "Rules & policies",
    icon: ShieldCheck,
    tone: "magenta",
    items: [
      "Can I sublet my apartment?",
      "How much notice must the landlord give to enter?",
      "How do I renew my lease?",
      "Is smoking allowed?",
    ],
  },
  {
    label: "Compare homes",
    icon: GitCompare,
    tone: "warm",
    items: [
      "Cheapest 2-bedroom apartment?",
      "Show me homes with a pool",
      "Which properties allow pets?",
      "Compare homes in Zilker",
    ],
  },
];

/** A leading TypeError from fetch means the backend is unreachable. */
function errText(e: unknown): string {
  return e instanceof TypeError
    ? "Could not reach the server. Is the backend running?"
    : e instanceof Error
      ? e.message
      : String(e);
}

/** Human label for the router's chosen path. */
function routeLabel(route: string): string {
  switch (route) {
    case "property":
      return "Property facts";
    case "lease":
      return "Lease terms";
    case "both":
      return "Property + lease";
    case "compare":
      return "Comparison";
    default:
      return "General";
  }
}

/** "Studio", "1 bd", "2 bd" — beds; 0 is a studio. */
function bedLabel(beds: number): string {
  return beds === 0 ? "Studio" : `${beds} bd`;
}

function CompareResults({
  items,
  onViewProperty,
  onAskAbout,
}: {
  items: CompareItem[];
  onViewProperty?: (id: string) => void;
  onAskAbout: (item: CompareItem) => void;
}) {
  return (
    <div style={{ marginTop: 12 }}>
      <div className="eyebrow">
        {items.length} matching home{items.length > 1 ? "s" : ""}
      </div>
      <div
        style={{ display: "flex", flexDirection: "column", gap: 8, marginTop: 8 }}
      >
        {items.map((item) => (
          <div
            key={item.id}
            className="subpanel"
            style={{
              display: "flex",
              alignItems: "flex-start",
              gap: 12,
              flexWrap: "wrap",
            }}
          >
            <div style={{ flex: "1 1 220px", minWidth: 0 }}>
              <div
                style={{
                  display: "flex",
                  alignItems: "baseline",
                  gap: 8,
                  flexWrap: "wrap",
                }}
              >
                <span style={{ fontWeight: 600 }}>{item.name}</span>
                <span
                  className="badge tone-info"
                  style={{ fontVariantNumeric: "tabular-nums" }}
                >
                  ${item.monthly_rent.toLocaleString()}/mo
                </span>
              </div>
              <div
                className="mini-row"
                style={{ justifyContent: "flex-start", gap: 12, marginTop: 4 }}
              >
                <span className="k">
                  {bedLabel(item.bedrooms)} · {item.bathrooms} ba
                </span>
                <span className="k">
                  {item.square_feet.toLocaleString()} sqft
                </span>
                <span className="k icon-line">
                  <MapPin size={12} /> {item.area}
                </span>
              </div>
              {item.matched.length > 0 && (
                <div className="chip-row" style={{ marginTop: 6 }}>
                  {item.matched.map((m) => (
                    <span key={m} className="chip" style={{ cursor: "default" }}>
                      {m}
                    </span>
                  ))}
                </div>
              )}
            </div>
            <div
              style={{ display: "flex", gap: 6, flexShrink: 0, flexWrap: "wrap" }}
            >
              {onViewProperty && (
                <button
                  className="btn-small btn-ghost"
                  onClick={() => onViewProperty(item.id)}
                  style={{ display: "inline-flex", alignItems: "center", gap: 6 }}
                >
                  <Building2 size={14} /> View
                </button>
              )}
              <button
                className="btn-small btn-ghost"
                onClick={() => onAskAbout(item)}
                style={{ display: "inline-flex", alignItems: "center", gap: 6 }}
              >
                <MessageSquare size={14} /> Ask about this home
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* memo()'d: patchMsg keeps unchanged messages at the same object reference,
 * so this skips re-rendering (and re-parsing Markdown for) every prior
 * transcript row on each streamed token of the message currently answering. */
const BotAnswer = memo(function BotAnswer({
  msg,
  onFollowUp,
  onOpenLease,
  onViewProperty,
  onAskAbout,
}: {
  msg: Msg;
  onFollowUp: (q: string) => void;
  onOpenLease: (target: LeaseTarget) => void;
  onViewProperty?: (id: string) => void;
  onAskAbout: (item: CompareItem) => void;
}) {
  const [showCites, setShowCites] = useState(false);
  const [copied, setCopied] = useState(false);
  const res = msg.res;
  const sources = res?.sources ?? [];
  const offline = res?.source === "rules";
  const followUps = res?.follow_ups ?? [];
  const comparison = res?.comparison ?? [];

  async function copyAnswer() {
    try {
      await navigator.clipboard.writeText(msg.text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable (permissions/insecure context) — ignore */
    }
  }

  // Still awaiting the first token: show the typing indicator only.
  if (msg.pending && !msg.text) {
    return (
      <div className="chat-msg bot" aria-label="Assistant is thinking">
        <span className="chat-typing">
          <span />
          <span />
          <span />
        </span>
      </div>
    );
  }

  return (
    <div className="chat-msg bot">
      <Markdown text={msg.text} />
      {res && (
        <>
          <div className="chat-actions">
            <span className="badge tone-info" title="How this answer was routed">
              {routeLabel(res.route)}
            </span>
            {offline && (
              <span
                className="badge tone-warn icon-line"
                title="Answered offline from deterministic rules (no LLM)"
              >
                <WifiOff size={12} /> Offline
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
            <button
              type="button"
              className="msg-copy"
              onClick={copyAnswer}
              title="Copy answer"
              aria-label="Copy answer"
            >
              {copied ? <Check size={13} /> : <Copy size={13} />}
            </button>
          </div>
          {showCites && sources.length > 0 && (
            <div className="cite-list">
              {sources.map((s, i) => {
                const clickable = !!(s.section && s.property_id);
                const body = (
                  <span>
                    <span className="concierge-cite-label">{s.label}</span>
                    {s.snippet && (
                      <span className="concierge-cite-snippet">{s.snippet}</span>
                    )}
                    {clickable && (
                      <span className="cite-view">View clause →</span>
                    )}
                  </span>
                );
                return clickable ? (
                  <button
                    type="button"
                    className="cite-item clickable"
                    key={i}
                    title={`Open the lease at "${s.section}"`}
                    onClick={() =>
                      onOpenLease({
                        propertyId: s.property_id!,
                        section: s.section,
                      })
                    }
                  >
                    <span className="cite-num">{s.cite ?? i + 1}</span>
                    {body}
                  </button>
                ) : (
                  <div className="cite-item" key={i}>
                    <span className="cite-num">{s.cite ?? i + 1}</span>
                    {body}
                  </div>
                );
              })}
            </div>
          )}
          {comparison.length > 0 && (
            <CompareResults
              items={comparison}
              onViewProperty={onViewProperty}
              onAskAbout={onAskAbout}
            />
          )}
          {followUps.length > 0 && (
            <>
              <div className="eyebrow" style={{ marginTop: 10 }}>
                Follow up
              </div>
              <div className="chip-row">
                {followUps.map((q) => (
                  <button
                    key={q}
                    className="chip"
                    onClick={() => onFollowUp(q)}
                  >
                    <Sparkles size={13} />
                    {q}
                  </button>
                ))}
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
});

export function Concierge({
  initialPropertyId,
  onViewProperty,
  health,
}: {
  initialPropertyId?: string;
  onViewProperty?: (id: string) => void;
  health?: Record<string, unknown> | null;
}) {
  const [properties, setProperties] = useState<Property[]>([]);
  const [propertyId, setPropertyId] = useState<string>(initialPropertyId ?? "");
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [lease, setLease] = useState<LeaseTarget | null>(null);
  const idRef = useRef(0);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Property catalog for the scope selector.
  useEffect(() => {
    getProperties()
      .then((res) => setProperties(res.properties))
      .catch(() => setProperties([]));
  }, []);

  // React to a new deep link coming from a property page.
  useEffect(() => {
    if (initialPropertyId) setPropertyId(initialPropertyId);
  }, [initialPropertyId]);

  // Keep the transcript pinned to the latest message, including mid-stream.
  // Skip while empty so the starter tiles land scrolled to the top, not
  // yanked to the bottom of their own content on first render.
  useEffect(() => {
    const el = scrollRef.current;
    if (el && messages.length > 0) el.scrollTop = el.scrollHeight;
  }, [messages]);

  /** Update a single message in place by id. */
  function patchMsg(id: number, fn: (m: Msg) => Msg) {
    setMessages((ms) => ms.map((m) => (m.id === id ? fn(m) : m)));
  }

  // useEvent: a permanently-stable identity so it can be passed to the
  // memo()'d BotAnswer rows without defeating their memoization, while still
  // always seeing the latest messages/propertyId/busy on every call.
  const send = useEvent(async (q: string, scopeOverride?: string) => {
    q = q.trim();
    const scope = scopeOverride ?? propertyId;
    if (!q || busy || !scope) return;
    // History is the conversation so far (before this turn), oldest first.
    const history = messages.map((m) => ({
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

    const request = {
      question: q,
      property_id: scope || undefined,
      history,
    };

    try {
      await conciergeAskStream(request, {
        onMeta: (meta) =>
          patchMsg(botId, (m) => ({
            ...m,
            res: {
              answer: m.text,
              route: meta.route,
              sources: meta.sources,
              source: "anthropic",
              property_id: meta.property_id,
              follow_ups: meta.follow_ups,
              comparison: meta.comparison,
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
        onDone: (source, sources) =>
          patchMsg(botId, (m) => ({
            ...m,
            pending: false,
            res: m.res
              ? { ...m.res, source, sources: sources ?? m.res.sources }
              : m.res,
          })),
      });
    } catch {
      // Streaming failed — fall back to the non-streaming endpoint.
      try {
        const res = await conciergeAsk(request);
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
      }
    } finally {
      setBusy(false);
    }
  });

  /** Scope to a compared home and ask about it in one tap. */
  const askAbout = useEvent((item: CompareItem) => {
    setPropertyId(item.id);
    send(`Tell me more about ${item.name}.`, item.id);
  });

  /** Start over: clear the transcript but keep the current property scope. */
  function clearChat() {
    setMessages([]);
  }

  const scopeName = properties.find((p) => p.id === propertyId)?.name;

  return (
    <div className="app app-ask">
      <header>
        <h1>Ask</h1>
        <p>
          Chat with a grounded assistant about any home's rent, amenities, and
          lease terms — every answer cites the actual listing or lease, and you
          can compare homes side by side.
        </p>
        <div className="badges">
          <TechBadge
            icon={Network}
            label="GraphRAG"
            title="Property facts come from the Neo4j graph; lease answers come from hybrid (vector + BM25) retrieval — both ground the Claude answer."
          />
          <TechBadge
            icon={Database}
            label="LlamaIndex"
            title="Every lease is chunked and indexed in Chroma via LlamaIndex; hybrid vector + BM25 retrieval (with FlashRank reranking) pulls the exact clauses that ground each answer."
          />
          <TechBadge
            icon={Link2}
            label="LangChain"
            title="The final answer is synthesized through a LangChain-wrapped Claude client."
          />
          <Badge on={!!health?.anthropic_key_set} label="Claude" tone="violet" />
          <Badge on={!!health?.neo4j_available} label="Neo4j" tone="blue" />
        </div>
      </header>

      <div className="card ask-chat-card">
        <div className="ask-chat-topbar">
          <div className={`ask-scope${scopeName ? " has-value" : ""}`}>
            <span className="ask-scope-icon">
              <MapPin size={13} aria-hidden />
            </span>
            <select
              className="ask-scope-select"
              value={propertyId}
              onChange={(e) => setPropertyId(e.target.value)}
              aria-label="Scope to a property"
            >
              <option value="" disabled>
                Select a property…
              </option>
              {properties.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name} · {p.area}
                </option>
              ))}
            </select>
            <ChevronDown size={14} className="ask-scope-chevron" aria-hidden />
          </div>
          <div style={{ marginLeft: "auto", display: "flex", gap: 6 }}>
            {scopeName && (
              <button
                type="button"
                className="icon-btn"
                onClick={() => setLease({ propertyId, tab: "document" })}
                title="View lease PDF"
                aria-label="View lease PDF"
              >
                <FileText size={16} />
              </button>
            )}
            {messages.length > 0 && (
              <button
                type="button"
                className="icon-btn"
                onClick={clearChat}
                title="Clear conversation"
                aria-label="Clear conversation"
              >
                <RotateCcw size={16} />
              </button>
            )}
          </div>
        </div>
        {!scopeName && (
          <p className="ask-scope-hint muted">
            Select a property above to ask questions about it.
          </p>
        )}
        <div className="ask-chat-scroll" ref={scrollRef} aria-live="polite">
          {messages.length === 0 ? (
            <>
              {scopeName && (
                <p className="muted" style={{ marginTop: 0 }}>
                  Pick a question to get started, or type your own below.
                </p>
              )}
              <div className="ask-starters">
                {STARTER_GROUPS.map((group) => (
                  <div
                    key={group.label}
                    className="ask-starter-group"
                    data-tone={group.tone}
                  >
                    <div className="eyebrow">{group.label}</div>
                    <div className="ask-starter-tiles">
                      {group.items.map((q) => (
                        <button
                          key={q}
                          type="button"
                          className="ask-starter-tile"
                          onClick={() => send(q)}
                          disabled={busy || !scopeName}
                        >
                          <group.icon size={13} />
                          <span>{q}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </>
          ) : (
            messages.map((m) =>
              m.who === "bot" ? (
                <div key={m.id} className="msg-row bot">
                  <span className="msg-avatar bot" aria-hidden>
                    <Sparkles size={13} />
                  </span>
                  <BotAnswer
                    msg={m}
                    onFollowUp={send}
                    onOpenLease={setLease}
                    onViewProperty={onViewProperty}
                    onAskAbout={askAbout}
                  />
                </div>
              ) : (
                <div key={m.id} className="msg-row user">
                  <div className="chat-msg user">{m.text}</div>
                  <span className="msg-avatar user" aria-hidden>
                    <User size={13} />
                  </span>
                </div>
              ),
            )
          )}
        </div>

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
              scopeName ? "e.g. What's the pet policy?" : "Select a property to ask…"
            }
            disabled={!scopeName}
          />
          <button type="submit" disabled={busy || !scopeName}>
            Ask
          </button>
        </form>
      </div>

      {lease && (
        <LeaseViewer
          propertyId={lease.propertyId}
          initialSection={lease.section}
          initialTab={lease.tab}
          onClose={() => setLease(null)}
        />
      )}
    </div>
  );
}
