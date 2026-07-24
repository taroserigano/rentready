import type { LucideIcon } from "lucide-react";

/**
 * Quiet, always-on badge naming a core technique behind this page — unlike
 * `Badge` (a live connected/not-connected service dot), this names an
 * architecture choice that doesn't have an up/down state of its own.
 */
export function TechBadge({
  icon: Icon,
  label,
  title,
  tone = "info",
}: {
  icon: LucideIcon;
  label: string;
  title: string;
  tone?: "info" | "warn";
}) {
  return (
    <span className={`badge icon-line tone-${tone}`} title={title}>
      <Icon size={12} aria-hidden />
      {label}
    </span>
  );
}
