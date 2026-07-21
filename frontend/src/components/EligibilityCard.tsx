import type { EligibilityResult } from "../types";
import { EligibilityGauge } from "./charts/EligibilityGauge";
import { Feedback } from "./Feedback";

const LABELS: Record<string, string> = {
  qualified: "Qualified",
  needs_review: "Needs review",
  not_qualified: "Not qualified",
};

export function EligibilityCard({
  result,
  applicantId,
}: {
  result: EligibilityResult;
  applicantId?: string;
}) {
  return (
    <div className="card">
      <h2>3. Eligibility</h2>
      <span className={`verdict ${result.verdict}`}>
        {LABELS[result.verdict]}
      </span>
      <EligibilityGauge ratio={result.income_to_rent_ratio} />
      {result.explanation && (
        <p style={{ fontSize: 14, marginTop: 12 }}>{result.explanation}</p>
      )}
      <ul className="reasons">
        {result.reasons.map((r, i) => (
          <li key={i}>{r}</li>
        ))}
      </ul>
      <div style={{ marginTop: 8 }}>
        <Feedback applicantId={applicantId} target="eligibility" />
      </div>
    </div>
  );
}
