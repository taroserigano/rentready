import { useState, type CSSProperties } from "react";
import { sendFeedback } from "../api";

export function Feedback({
  applicantId,
  target,
  itemId,
}: {
  applicantId?: string;
  target: string;
  itemId?: string;
}) {
  const [sent, setSent] = useState<"up" | "down" | "">("");

  async function rate(rating: "up" | "down") {
    setSent(rating);
    try {
      await sendFeedback({ applicant_id: applicantId, target, rating, item_id: itemId });
    } catch {
      setSent("");
    }
  }

  if (sent) {
    return (
      <span style={{ fontSize: 11, color: "var(--muted)" }}>
        Thanks for the feedback {sent === "up" ? "👍" : "👎"}
      </span>
    );
  }

  return (
    <span style={{ display: "inline-flex", gap: 6, alignItems: "center" }}>
      <span style={{ fontSize: 11, color: "var(--muted)" }}>Helpful?</span>
      <button
        onClick={() => rate("up")}
        title="Helpful"
        style={btn}
      >
        👍
      </button>
      <button
        onClick={() => rate("down")}
        title="Not helpful"
        style={btn}
      >
        👎
      </button>
    </span>
  );
}

const btn: CSSProperties = {
  background: "transparent",
  border: "1px solid var(--line-strong)",
  borderRadius: 8,
  padding: "2px 8px",
  cursor: "pointer",
  fontSize: 12,
  boxShadow: "none",
};
