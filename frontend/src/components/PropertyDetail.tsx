import { useEffect, useRef, type ReactNode } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Bath,
  BedDouble,
  CalendarDays,
  Car,
  ExternalLink,
  Footprints,
  LayoutGrid,
  MapPin,
  PawPrint,
  Plug,
  Ruler,
  ShieldCheck,
  Sofa,
  Sun,
  TrainFront,
  WashingMachine,
  Wrench,
  X,
} from "lucide-react";
import type { FloorPlan, Property } from "../types";
import { PropThumb } from "./PropPhoto";

export const PARKING_LABELS: Record<string, string> = {
  garage: "Garage parking",
  covered: "Covered parking",
  lot: "Lot parking",
  surface: "Lot parking",
  street: "Street parking",
  none: "No parking",
};

export const LAUNDRY_LABELS: Record<string, string> = {
  in_unit: "In-unit washer & dryer",
  hookups: "Washer/dryer hookups",
  shared: "Shared laundry room",
  none: "No laundry on site",
};

/** True only for a real, finite number — guards against missing fields. */
export function isNum(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

export function money(n: number): string {
  return `$${n.toLocaleString()}`;
}

/** "2026-08-01" -> "Aug 1". Returns undefined for missing/bad dates. */
export function shortDate(iso: unknown): string | undefined {
  if (typeof iso !== "string" || !/^\d{4}-\d{2}-\d{2}/.test(iso)) return undefined;
  const d = new Date(`${iso.slice(0, 10)}T00:00:00`);
  if (Number.isNaN(d.getTime())) return undefined;
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function cap(s: unknown): string | undefined {
  if (typeof s !== "string" || !s) return undefined;
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/** Rent per square foot, e.g. "$2.10". Undefined when sqft is 0/missing. */
export function pricePerSqft(rent: unknown, sqft: unknown): number | undefined {
  if (!isNum(rent) || !isNum(sqft) || sqft <= 0) return undefined;
  return Math.round((rent / sqft) * 100) / 100;
}

/** Total open units across floor plans (falls back to unit_count). */
export function totalAvailableUnits(p: {
  floor_plans?: FloorPlan[];
  unit_count?: number;
}): number | undefined {
  const plans = Array.isArray(p.floor_plans) ? p.floor_plans : [];
  const withUnits = plans.filter((fp) => isNum(fp?.available_units));
  if (withUnits.length > 0)
    return withUnits.reduce((sum, fp) => sum + (fp.available_units as number), 0);
  return undefined;
}

/** True when the earliest availability date is on/before today. */
export function isAvailableNow(p: {
  availability_date?: string;
  floor_plans?: FloorPlan[];
}): boolean {
  const dates = [
    p.availability_date,
    ...(Array.isArray(p.floor_plans)
      ? p.floor_plans.map((fp) => fp?.availability_date)
      : []),
  ].filter((d): d is string => typeof d === "string" && /^\d{4}-\d{2}-\d{2}/.test(d));
  if (dates.length === 0) return false;
  const earliest = dates.sort()[0].slice(0, 10);
  const today = new Date().toISOString().slice(0, 10);
  return earliest <= today;
}

/* ---- floor-plan helpers (shared by browser cards, page, and modal) ---- */

/** "3" or "2.5" — no trailing ".0" on whole numbers. */
function fmtPlanNum(n: number): string {
  return Number.isInteger(n) ? String(n) : n.toFixed(1);
}

/**
 * The property's usable floor plans, cheapest first.
 * Returns [] when the field is missing or malformed (older cached rows,
 * or the backend hasn't shipped floor_plans yet) — callers then fall
 * back to the top-level single-unit fields.
 */
export function sortedPlans(p: { floor_plans?: FloorPlan[] }): FloorPlan[] {
  if (!Array.isArray(p.floor_plans)) return [];
  return p.floor_plans
    .filter(
      (fp): fp is FloorPlan =>
        !!fp &&
        typeof fp === "object" &&
        isNum(fp.monthly_rent) &&
        isNum(fp.bedrooms) &&
        isNum(fp.square_feet),
    )
    .sort((a, b) => a.monthly_rent - b.monthly_rent);
}

/** "Studio" or "2 bd" — 0 bedrooms means studio. */
export function bedShort(bedrooms: number): string {
  return bedrooms === 0 ? "Studio" : `${bedrooms} bd`;
}

/** Bed range across plans: "Studio – 3 bd", "1 – 2 bd", or just "2 bd". */
export function bedRangeLabel(plans: FloorPlan[]): string {
  const beds = plans.map((fp) => fp.bedrooms);
  const min = Math.min(...beds);
  const max = Math.max(...beds);
  if (min === max) return bedShort(min);
  return min === 0 ? `Studio – ${max} bd` : `${min} – ${max} bd`;
}

/** Bath range: "1 – 2.5 bath" or "2 bath". Undefined when no plan has baths. */
export function bathRangeLabel(plans: FloorPlan[]): string | undefined {
  const baths = plans.map((fp) => fp.bathrooms).filter(isNum);
  if (baths.length === 0) return undefined;
  const min = Math.min(...baths);
  const max = Math.max(...baths);
  return min === max
    ? `${fmtPlanNum(min)} bath`
    : `${fmtPlanNum(min)} – ${fmtPlanNum(max)} bath`;
}

/** Size range: "450 – 1,200 sq ft" or "800 sq ft". */
export function sqftRangeLabel(plans: FloorPlan[]): string {
  const sizes = plans.map((fp) => fp.square_feet);
  const min = Math.min(...sizes);
  const max = Math.max(...sizes);
  return min === max
    ? `${min.toLocaleString()} sq ft`
    : `${min.toLocaleString()} – ${max.toLocaleString()} sq ft`;
}

function formatBaths(p: Property): string {
  const n = Number.isInteger(p.bathrooms)
    ? String(p.bathrooms)
    : p.bathrooms.toFixed(1);
  const type = p.bathroom_type === "shower_only" ? "shower only" : "full bath";
  return `${n} bath (${type})`;
}

interface DetailRow {
  k: string;
  v: string;
}

/** Section of label/value rows. Renders nothing when there are no rows. */
function Section({
  title,
  icon,
  rows,
  children,
}: {
  title: string;
  icon: ReactNode;
  rows?: DetailRow[];
  children?: ReactNode;
}) {
  const hasRows = (rows?.length ?? 0) > 0;
  if (!hasRows && !children) return null;
  return (
    <div className="detail-section">
      <h3 className="icon-line">
        {icon}
        {title}
      </h3>
      {hasRows && (
        <div className="detail-rows">
          {rows!.map((r) => (
            <div className="detail-row" key={r.k}>
              <span className="k">{r.k}</span>
              <span>{r.v}</span>
            </div>
          ))}
        </div>
      )}
      {children}
    </div>
  );
}

/**
 * One compact line per plan for the quick-look modal, e.g.
 * "1 Bedroom — 1 bd · 1 bath · 620 sq ft · $1,450/mo · 3 available".
 */
function planRows(plans: FloorPlan[]): DetailRow[] {
  return plans.map((fp, i) => ({
    k: fp.name || `Plan ${i + 1}`,
    v: [
      bedShort(fp.bedrooms),
      isNum(fp.bathrooms) ? `${fmtPlanNum(fp.bathrooms)} bath` : null,
      `${fp.square_feet.toLocaleString()} sq ft`,
      `${money(fp.monthly_rent)}/mo`,
      isNum(fp.available_units)
        ? fp.available_units > 0
          ? `${fp.available_units} available`
          : "None open"
        : null,
    ]
      .filter(Boolean)
      .join(" · "),
  }));
}

function costRows(p: Property): DetailRow[] {
  const out: DetailRow[] = [{ k: "Monthly rent", v: `${money(p.monthly_rent)}/mo` }];
  if (isNum(p.security_deposit))
    out.push({ k: "Security deposit", v: money(p.security_deposit) });
  if (isNum(p.application_fee))
    out.push({ k: "Application fee", v: money(p.application_fee) });
  if (isNum(p.admin_fee)) out.push({ k: "Admin fee", v: money(p.admin_fee) });
  if (isNum(p.parking_fee_monthly) && p.parking_fee_monthly > 0)
    out.push({ k: "Parking", v: `${money(p.parking_fee_monthly)}/mo` });
  if (isNum(p.pet_deposit) && p.pet_deposit > 0)
    out.push({ k: "Pet deposit", v: money(p.pet_deposit) });
  if (isNum(p.pet_rent_monthly) && p.pet_rent_monthly > 0)
    out.push({ k: "Pet rent", v: `${money(p.pet_rent_monthly)}/mo` });
  return out;
}

function requirementRows(p: Property): DetailRow[] {
  const out: DetailRow[] = [];
  if (isNum(p.min_credit_score))
    out.push({ k: "Min credit score", v: String(p.min_credit_score) });
  if (isNum(p.min_income_multiplier))
    out.push({ k: "Income needed", v: `${p.min_income_multiplier}× rent` });
  if (isNum(p.max_occupants))
    out.push({ k: "Max occupants", v: String(p.max_occupants) });
  if (typeof p.smoking_allowed === "boolean")
    out.push({ k: "Smoking", v: p.smoking_allowed ? "Allowed" : "Not allowed" });
  return out;
}

function homeRows(p: Property): DetailRow[] {
  const out: DetailRow[] = [];
  if (isNum(p.year_built)) out.push({ k: "Year built", v: String(p.year_built) });
  const avail = shortDate(p.availability_date);
  if (avail) out.push({ k: "Available", v: avail });
  const flooring = cap(p.flooring);
  if (flooring) out.push({ k: "Flooring", v: flooring });
  const heating = cap(p.heating);
  if (heating)
    out.push({ k: "Heating", v: heating === "None" ? "None" : `${heating} heat` });
  const cooling = cap(p.cooling);
  if (cooling)
    out.push({ k: "Cooling", v: cooling === "None" ? "None" : `${cooling} AC` });
  if (isNum(p.floor_level)) out.push({ k: "Floor", v: `Level ${p.floor_level}` });
  const laundry = p.laundry_type ? LAUNDRY_LABELS[p.laundry_type] : undefined;
  if (laundry) out.push({ k: "Laundry", v: laundry });
  if (isNum(p.unit_count)) out.push({ k: "Units in building", v: String(p.unit_count) });
  return out;
}

interface DetailProps {
  property: Property | null;
  onClose: () => void;
  /** When set, shows a "View full listing" button that opens the property page. */
  onViewListing?: (id: string) => void;
}

export function PropertyDetail({ property, onClose, onViewListing }: DetailProps) {
  const plans = property ? sortedPlans(property) : [];
  const multiPlan = plans.length > 1;

  // Close on Escape while the modal is open.
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!property) return;
    const prevFocus = document.activeElement as HTMLElement | null;
    // Move focus into the dialog on open.
    panelRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
        return;
      }
      if (e.key === "Tab" && panelRef.current) {
        // Trap focus within the panel.
        const els = panelRef.current.querySelectorAll<HTMLElement>(
          'a[href], button:not([disabled]), input, select, textarea, [tabindex]:not([tabindex="-1"])',
        );
        if (els.length === 0) return;
        const first = els[0];
        const last = els[els.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      prevFocus?.focus?.(); // restore focus to the trigger on close
    };
  }, [property, onClose]);

  return (
    <AnimatePresence>
      {property && (
        <motion.div
          className="modal-backdrop"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
          onClick={onClose}
        >
          <motion.div
            className="modal-panel"
            ref={panelRef}
            tabIndex={-1}
            role="dialog"
            aria-modal="true"
            aria-label={property.name}
            initial={{ opacity: 0, scale: 0.96 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.96 }}
            transition={{ duration: 0.18, ease: "easeOut" }}
            onClick={(e) => e.stopPropagation()}
          >
            <PropThumb
              src={property.photo_url || property.photo_urls?.[0]}
              alt={property.name}
              className="modal-img"
              phClassName="modal-img prop-img-ph"
            />
            <div className="modal-body">
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "flex-start",
                  gap: 12,
                }}
              >
                <div>
                  <div style={{ fontWeight: 700, fontSize: 17 }}>
                    {property.name}
                  </div>
                  <div className="icon-line muted" style={{ fontSize: 13 }}>
                    <MapPin size={14} />
                    {property.city
                      ? `${property.area} · ${property.city}`
                      : property.area}
                  </div>
                </div>
                <div style={{ display: "flex", gap: 8, flexShrink: 0 }}>
                  {onViewListing && (
                    <button
                      className="btn-small"
                      onClick={() => onViewListing(property.id)}
                      style={{ display: "inline-flex", alignItems: "center", gap: 6 }}
                    >
                      <ExternalLink size={14} /> View full listing
                    </button>
                  )}
                  <button
                    className="btn-small btn-ghost"
                    onClick={onClose}
                    aria-label="Close"
                    style={{ display: "inline-flex", alignItems: "center", gap: 6 }}
                  >
                    <X size={14} /> Close
                  </button>
                </div>
              </div>

              <div className="modal-rent">
                {multiPlan && <span>From </span>}
                ${property.monthly_rent.toLocaleString()}
                <span> / month</span>
              </div>

              <div className="spec-grid">
                <div className="spec-item">
                  <BedDouble size={15} />
                  {multiPlan ? bedRangeLabel(plans) : `${property.bedrooms} bed`}
                </div>
                <div className="spec-item">
                  <Bath size={15} />
                  {multiPlan
                    ? (bathRangeLabel(plans) ?? formatBaths(property))
                    : formatBaths(property)}
                </div>
                <div className="spec-item">
                  <Ruler size={15} />
                  {multiPlan
                    ? sqftRangeLabel(plans)
                    : `${property.square_feet.toLocaleString()} sq ft`}
                </div>
                <div className="spec-item">
                  <Car size={15} />
                  {PARKING_LABELS[property.parking_type] ?? "Parking unknown"}
                </div>
                <div className="spec-item">
                  <CalendarDays size={15} />
                  {property.lease_term_months}-month lease
                </div>
                <div className="spec-item">
                  <Sofa size={15} />
                  {property.furnished ? "Furnished" : "Unfurnished"}
                </div>
              </div>

              <div className="badges" style={{ marginTop: 12 }}>
                {property.pets_allowed && (
                  <span className="badge tone-good icon-line">
                    <PawPrint size={12} /> Pets OK
                  </span>
                )}
                {property.in_unit_laundry && (
                  <span className="badge tone-info icon-line">
                    <WashingMachine size={12} /> In-unit laundry
                  </span>
                )}
                {property.has_balcony && (
                  <span className="badge tone-info icon-line">
                    <Sun size={12} /> Balcony
                  </span>
                )}
                {property.gated === true && (
                  <span className="badge tone-info">Gated community</span>
                )}
                {property.storage_unit_available === true && (
                  <span className="badge tone-info">Storage available</span>
                )}
                {property.amenities.map((a) => (
                  <span key={a} className="badge tone-info">
                    {a}
                  </span>
                ))}
              </div>

              {multiPlan && (
                <Section
                  title="Floor plans"
                  icon={<LayoutGrid size={13} />}
                  rows={planRows(plans)}
                />
              )}

              <Section
                title="Costs & fees"
                icon={<Wrench size={13} />}
                rows={costRows(property)}
              />

              <Section
                title="Requirements"
                icon={<ShieldCheck size={13} />}
                rows={requirementRows(property)}
              />

              <Section
                title="Home details"
                icon={<Sofa size={13} />}
                rows={homeRows(property)}
              >
                {(property.appliances?.length ?? 0) > 0 && (
                  <div className="badges" style={{ marginTop: 8 }}>
                    {property.appliances!.map((a) => (
                      <span key={a} className="badge tone-info">
                        {a}
                      </span>
                    ))}
                  </div>
                )}
              </Section>

              {(property.utilities_included?.length ?? 0) > 0 && (
                <Section title="Utilities included" icon={<Plug size={13} />}>
                  <div className="badges">
                    {property.utilities_included!.map((u) => (
                      <span key={u} className="badge tone-good">
                        {u}
                      </span>
                    ))}
                  </div>
                </Section>
              )}

              {(property.walk_score != null ||
                property.transit_score != null) && (
                <div
                  className="muted"
                  style={{
                    marginTop: 14,
                    display: "flex",
                    gap: 16,
                    fontSize: 13,
                  }}
                >
                  {property.walk_score != null && (
                    <span className="icon-line">
                      <Footprints size={14} /> Walk score {property.walk_score}
                    </span>
                  )}
                  {property.transit_score != null && (
                    <span className="icon-line">
                      <TrainFront size={14} /> Transit score{" "}
                      {property.transit_score}
                    </span>
                  )}
                </div>
              )}

              <div className="muted" style={{ marginTop: 10, fontSize: 12 }}>
                {property.property_type} · listing {property.id}
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
