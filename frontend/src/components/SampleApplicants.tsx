import { useEffect, useState } from "react";
import { getSamples, type Sample } from "../api";

export function SampleApplicants({
  onPick,
  disabled,
}: {
  onPick: (slug: string) => void;
  disabled: boolean;
}) {
  const [samples, setSamples] = useState<Sample[]>([]);

  useEffect(() => {
    getSamples()
      .then(setSamples)
      .catch(() => setSamples([]));
  }, []);

  if (samples.length === 0) return null;

  return (
    <div style={{ marginTop: 16 }}>
      <div className="eyebrow" style={{ marginBottom: 10 }}>
        Or try a sample
      </div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        {samples.map((s) => (
          <button
            key={s.slug}
            className="btn-ghost btn-small sample-chip"
            onClick={() => onPick(s.slug)}
            disabled={disabled}
          >
            {s.name}
          </button>
        ))}
      </div>
    </div>
  );
}
