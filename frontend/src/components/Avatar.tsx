/**
 * Offline-safe applicant avatar: initials on a deterministic gradient derived
 * from the name. Pure inline SVG — no network, no external avatar service, so
 * it always renders (matches the app's offline-degradation posture).
 */

function hash(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
  return Math.abs(h);
}

function initials(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "?";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

export function Avatar({
  name,
  size = 40,
  className = "",
}: {
  name?: string | null;
  size?: number;
  className?: string;
}) {
  const label = (name || "").trim() || "Unknown";
  const h = hash(label);
  const hue = h % 360;
  const hue2 = (hue + 40) % 360;
  const from = `hsl(${hue}, 62%, 52%)`;
  const to = `hsl(${hue2}, 58%, 42%)`;
  const gid = `av${h}`;
  return (
    <svg
      className={`avatar ${className}`}
      width={size}
      height={size}
      viewBox="0 0 40 40"
      role="img"
      aria-label={label}
    >
      <defs>
        <linearGradient id={gid} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor={from} />
          <stop offset="100%" stopColor={to} />
        </linearGradient>
      </defs>
      <rect width="40" height="40" rx="20" fill={`url(#${gid})`} />
      <text
        x="20"
        y="21"
        dominantBaseline="central"
        textAnchor="middle"
        fontSize="15"
        fontWeight="600"
        fill="#fff"
        fontFamily="inherit"
      >
        {initials(label)}
      </text>
    </svg>
  );
}
