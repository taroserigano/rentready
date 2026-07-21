import { Check, Loader2 } from "lucide-react";

export type Phase = "idle" | "extracting" | "screening" | "done" | "error";

type StepState = "pending" | "active" | "done" | "error";

function stepState(phase: Phase, step: "extract" | "eligibility" | "match", ready: {
  profile: boolean;
  eligibility: boolean;
  recs: boolean;
}): StepState {
  if (phase === "error") {
    // Mark completed steps done; the first not-ready step is where it failed.
    if (step === "extract") return ready.profile ? "done" : "error";
    if (step === "eligibility")
      return ready.eligibility ? "done" : ready.profile ? "error" : "pending";
    return ready.recs ? "done" : ready.eligibility ? "error" : "pending";
  }
  if (step === "extract")
    return ready.profile ? "done" : phase === "extracting" ? "active" : "pending";
  if (step === "eligibility")
    return ready.eligibility ? "done" : phase === "screening" ? "active" : "pending";
  return ready.recs ? "done" : phase === "screening" ? "active" : "pending";
}

function Dot({ state }: { state: StepState }) {
  return (
    <span className="step-dot">
      {state === "done" ? (
        <Check size={13} />
      ) : state === "active" ? (
        <Loader2 size={13} className="spin" />
      ) : state === "error" ? (
        "!"
      ) : (
        <span style={{ width: 6, height: 6, borderRadius: 999, background: "currentColor" }} />
      )}
    </span>
  );
}

/** Horizontal pipeline: Extract profile → Check eligibility → Match properties. */
export function Stepper({
  phase,
  ready,
}: {
  phase: Phase;
  ready: { profile: boolean; eligibility: boolean; recs: boolean };
}) {
  if (phase === "idle" || phase === "done") return null;
  const steps: Array<{ key: "extract" | "eligibility" | "match"; label: string }> = [
    { key: "extract", label: "Extract profile" },
    { key: "eligibility", label: "Check eligibility" },
    { key: "match", label: "Match properties" },
  ];
  return (
    <div className="stepper" role="status" aria-live="polite">
      {steps.map((s, i) => {
        const st = stepState(phase, s.key, ready);
        return (
          <span key={s.key} style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
            <span className={`step ${st}`} aria-current={st === "active" ? "step" : undefined}>
              <Dot state={st} />
              <span className="step-label">{s.label}</span>
            </span>
            {i < steps.length - 1 && (
              <span className={`step-line${st === "done" ? " filled" : ""}`} />
            )}
          </span>
        );
      })}
    </div>
  );
}

/** A card of shimmer placeholders that occupies the target card's shape. */
export function SkeletonCard({
  title,
  lines = 3,
  block,
}: {
  title: string;
  lines?: number;
  block?: number;
}) {
  return (
    <div className="card" aria-hidden>
      <h2 style={{ opacity: 0.5 }}>{title}</h2>
      {Array.from({ length: lines }).map((_, i) => (
        <div key={i} className={`skel skel-line${i === lines - 1 ? " sm" : ""}`} />
      ))}
      {block ? <div className="skel skel-block" style={{ height: block, marginTop: 10 }} /> : null}
    </div>
  );
}
