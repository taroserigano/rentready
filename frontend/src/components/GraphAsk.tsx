import { useState } from "react";
import { Check, Copy, Sparkles } from "lucide-react";
import { graphAsk } from "../api";
import { Markdown } from "./Markdown";

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
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);

  async function run(question: string) {
    if (busy) return;
    setBusy(true);
    setAnswer("");
    setCypher("");
    try {
      const res = await graphAsk(question);
      setAnswer(res.answer);
      setCypher(res.cypher);
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

      {answer && <div className="chat-msg bot"><Markdown text={answer} /></div>}
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
