import { Sparkles, WifiOff } from "lucide-react";
import type { RiskChatAnswer, RiskChatIntent } from "../../types";
import { RiskChatArtifact } from "./RiskChatArtifact";

/** One message in the risk-chat thread. */
export interface RiskChatMsg {
  /** Stable id so async updates can target this exact message. */
  id: number;
  who: "user" | "bot" | "system";
  text: string;
  res?: RiskChatAnswer;
  /** Bot message awaiting its answer (shows the typing dots). */
  pending?: boolean;
}

/** Human label for the router's chosen intent. */
function intentLabel(intent: RiskChatIntent): string {
  switch (intent) {
    case "explain":
      return "Explanation";
    case "whatif":
      return "What-if";
    case "counterfactual":
      return "Path to lower risk";
    case "compare":
      return "Comparison";
    case "exclusions":
      return "Model governance";
    default:
      return "General";
  }
}

/** These intents are always exploratory — never the saved score. */
function isExploratory(intent: RiskChatIntent): boolean {
  return intent === "whatif" || intent === "counterfactual";
}

export function RiskChatMessage({
  msg,
  onFollowUp,
  onSelectApplicant,
}: {
  msg: RiskChatMsg;
  onFollowUp: (q: string) => void;
  /** Lets a comparison peer row re-point the page's selected applicant. */
  onSelectApplicant?: (id: string) => void;
}) {
  const res = msg.res;
  const offline = res?.source === "rules";
  const followUps = res?.follow_ups ?? [];
  const artifact = res?.artifact ?? { kind: "none" as const };

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
              {intentLabel(res.intent)}
            </span>
            {isExploratory(res.intent) && (
              <span className="badge tone-warn">
                Exploratory — not the saved score
              </span>
            )}
            {offline && (
              <span
                className="badge tone-warn icon-line"
                title="Answered offline from deterministic rules (no LLM)"
              >
                <WifiOff size={12} /> Offline
              </span>
            )}
          </div>

          <RiskChatArtifact
            artifact={artifact}
            onSelectApplicant={onSelectApplicant}
          />


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
}
