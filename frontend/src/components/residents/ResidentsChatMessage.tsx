import { Sparkles, WifiOff } from "lucide-react";
import type { ResidentChatAnswer } from "../../types";
import { ResidentsChatArtifact } from "./ResidentsChatArtifact";

/** One message in the residents-chat thread. */
export interface ResidentsChatMsg {
  /** Stable id so async updates can target this exact message. */
  id: number;
  who: "user" | "bot" | "system";
  text: string;
  res?: ResidentChatAnswer;
  /** Bot message awaiting its answer (shows the typing dots). */
  pending?: boolean;
}

/** Human label for the router's chosen intent. */
function intentLabel(intent: string): string {
  switch (intent) {
    case "explain":
      return "Explanation";
    case "horizon":
      return "Late-payment outlook";
    case "frequency":
      return "How often";
    case "severity":
      return "How severe";
    case "arrears":
      return "Arrears";
    case "cure":
      return "Getting current";
    case "retention":
      return "Retention";
    case "property_health":
      return "Property health";
    case "compare":
      return "Comparison";
    case "governance":
      return "Model governance";
    default:
      return "General";
  }
}

export function ResidentsChatMessage({
  msg,
  onFollowUp,
  onSelectProperty,
}: {
  msg: ResidentsChatMsg;
  onFollowUp: (q: string) => void;
  onSelectProperty?: (propertyId: string) => void;
}) {
  const res = msg.res;
  const offline = res?.source === "rules";
  const followUps = res?.follow_ups ?? [];

  // Still awaiting the answer: show the typing indicator only.
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
      {msg.text}
      {res && (
        <>
          <div className="chat-actions">
            <span className="badge tone-info" title="How this answer was routed">
              {intentLabel(String(res.intent))}
            </span>
            {offline && (
              <span
                className="badge tone-warn icon-line"
                title="Answered offline from deterministic rules (no LLM)"
              >
                <WifiOff size={12} /> Offline
              </span>
            )}
          </div>

          <ResidentsChatArtifact
            artifact={res.artifact}
            onSelectProperty={onSelectProperty}
          />

          {followUps.length > 0 && (
            <>
              <div className="eyebrow" style={{ marginTop: 10 }}>
                Follow up
              </div>
              <div className="chip-row">
                {followUps.map((q) => (
                  <button key={q} className="chip" onClick={() => onFollowUp(q)}>
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
}
