/** Quiet service-health chip: neutral outline, colored status dot. */
export function Badge({
  on,
  label,
  tone,
}: {
  on: boolean;
  label: string;
  tone?: "violet" | "teal" | "magenta" | "blue";
}) {
  return (
    <span
      className="badge tone-info"
      data-tone={tone}
      title={`${label} is ${on ? "connected" : "not connected"}`}
    >
      <span className={`dot ${on ? "good" : "bad"}`} aria-hidden />
      {label}
      <span className="sr-only">{on ? "connected" : "not connected"}</span>
    </span>
  );
}
