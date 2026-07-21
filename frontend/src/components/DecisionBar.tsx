import { useEffect, useState } from "react";
import { Check, Clock, HelpCircle, X } from "lucide-react";
import { getDecisions, postDecision } from "../api";
import type { DecisionItem } from "../types";
import { toast } from "../toast";

export const STATUS_TONE: Record<string, string> = {
  approve: "good",
  decline: "bad",
  waitlist: "warn",
  request_info: "info",
  new: "info",
};

export const STATUS_LABEL: Record<string, string> = {
  approve: "Approved",
  decline: "Declined",
  waitlist: "Waitlisted",
  request_info: "Info requested",
  new: "New",
};

const ACTIONS: Array<{ id: string; label: string; icon: JSX.Element }> = [
  { id: "approve", label: "Approve", icon: <Check size={14} /> },
  { id: "decline", label: "Decline", icon: <X size={14} /> },
  { id: "waitlist", label: "Waitlist", icon: <Clock size={14} /> },
  { id: "request_info", label: "Request info", icon: <HelpCircle size={14} /> },
];

/** Reviewer decision actions + note + per-applicant audit trail (F6). */
export function DecisionBar({
  applicantId,
  reviewer,
  onDecision,
}: {
  applicantId: string;
  reviewer?: string;
  onDecision?: (status: string) => void;
}) {
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState("");
  const [history, setHistory] = useState<DecisionItem[]>([]);

  const load = () =>
    getDecisions(applicantId)
      .then((d) => setHistory(d.decisions))
      .catch(() => {});

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [applicantId]);

  async function decide(action: string) {
    setBusy(action);
    try {
      await postDecision(applicantId, { action, note: note.trim(), reviewer });
      setNote("");
      await load();
      toast(`Marked ${STATUS_LABEL[action] ?? action}`, "good");
      onDecision?.(action);
    } catch {
      toast("Couldn't save the decision", "bad");
    } finally {
      setBusy("");
    }
  }

  const current = history[0];

  return (
    <div className="card">
      <div className="rec-head">
        <h2 style={{ margin: 0 }}>Decision</h2>
        {current && (
          <span
            className={`badge tone-${STATUS_TONE[current.action] ?? "info"}`}
            style={{ marginLeft: "auto" }}
          >
            {STATUS_LABEL[current.action] ?? current.action}
          </span>
        )}
      </div>
      <input
        className="field"
        style={{ marginTop: 8 }}
        placeholder="Add a note (optional)…"
        value={note}
        onChange={(e) => setNote(e.target.value)}
      />
      <div className="chip-row" style={{ marginTop: 10 }}>
        {ACTIONS.map((a) => (
          <button
            key={a.id}
            className="chip"
            disabled={!!busy}
            onClick={() => decide(a.id)}
          >
            {a.icon}
            {busy === a.id ? "Saving…" : a.label}
          </button>
        ))}
      </div>
      {history.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <div className="muted" style={{ fontSize: 12, marginBottom: 4 }}>
            History
          </div>
          {history.map((h, i) => (
            <div
              key={i}
              className="icon-line"
              style={{ fontSize: 12, color: "var(--text-2)", padding: "3px 0" }}
            >
              <span className={`badge tone-${STATUS_TONE[h.action] ?? "info"}`}>
                {STATUS_LABEL[h.action] ?? h.action}
              </span>
              {h.reviewer ? ` · ${h.reviewer}` : ""}
              {h.note ? ` · “${h.note}”` : ""}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
