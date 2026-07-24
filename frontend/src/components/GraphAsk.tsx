import { useState } from "react";
import { Check, Copy, Sparkles, Zap } from "lucide-react";
import { graphAsk } from "../api";
import { Markdown } from "./Markdown";

/** How the last answer was produced. "template" = a deterministic Cypher
 * template matched (no Cypher-writing LLM call); "anthropic" = Claude wrote
 * the Cypher; "rules" = degraded (offline / error). */
type AskSource = "template" | "anthropic" | "rules";

const EXAMPLES = [
  "Which properties in South Congress allow pets?",
  "Cheapest 2-bedroom under $2,000?",
  "Which areas have a gym?",
  "Homes with a pool and parking",
];

export function GraphAsk({ neo4jAvailable }: { neo4jAvailable?: boolean }) {
  const [q, setQ] = useState(EXAMPLES[0]);
  const [answer, setAnswer] = useState("");
  const [cypher, setCypher] = useState("");
  const [source, setSource] = useState<AskSource | null>(null);
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);

  async function run(question: string) {
    if (busy) return;
    setBusy(true);
    setAnswer("");
    setCypher("");
    setSource(null);
    try {
      const res = await graphAsk(question);
      setAnswer(res.answer);
      setCypher(res.cypher);
      setSource(res.source);
    } catch (err) {
      setAnswer(`Error: ${err}`);
    } finally {
      setBusy(false);
    }
  }

  async function copyCypher() {
    try {
      await navigator.clipboard.writeText(cypher);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable (insecure context) — no-op */
    }
  }

  const offline = neo4jAvailable === false;

  return (
    <div className="card">
      <h2>6. Ask the property graph (langchain-neo4j)</h2>
      <p className="muted">
        Claude writes a Cypher query against Neo4j and answers from the result.
      </p>

      {offline ? (
        <div className="alert bad" role="status" style={{ marginTop: 12 }}>
          Graph search is offline — connect Neo4j to query the property graph.
        </div>
      ) : (
        <>
          <div className="chip-row">
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                className="chip"
                disabled={busy}
                onClick={() => {
                  setQ(ex);
                  run(ex);
                }}
              >
                <Sparkles size={13} />
                {ex}
              </button>
            ))}
          </div>
          <form
            className="chat-form"
            onSubmit={(e) => {
              e.preventDefault();
              run(q);
            }}
          >
            <input value={q} onChange={(e) => setQ(e.target.value)} />
            <button type="submit" disabled={busy}>
              {busy ? "…" : "Query"}
            </button>
          </form>
        </>
      )}

      {answer && (
        <div className="chat-msg bot">
          <Markdown text={answer} />
          {source === "template" && (
            <div className="chat-actions">
              <span
                className="badge tone-good icon-line"
                title="Matched a deterministic Cypher template — no Cypher-writing LLM call needed"
              >
                <Zap size={12} /> Template
              </span>
            </div>
          )}
          {source === "anthropic" && (
            <div className="chat-actions">
              <span
                className="badge tone-info icon-line"
                title="No template matched this question — Claude wrote the Cypher"
              >
                <Sparkles size={12} /> Claude-written Cypher
              </span>
            </div>
          )}
        </div>
      )}
      {cypher && (
        <div className="code-wrap">
          <pre className="code-block">{cypher}</pre>
          <button
            className={`code-copy${copied ? " copied" : ""}`}
            onClick={copyCypher}
            aria-label="Copy Cypher query"
            title="Copy Cypher"
          >
            {copied ? <Check size={13} /> : <Copy size={13} />}
          </button>
        </div>
      )}
    </div>
  );
}
