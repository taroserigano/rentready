import { useEffect, useState, type ReactNode } from "react";
import { motion } from "framer-motion";
import {
  ArrowLeft,
  Banknote,
  Bath,
  BedDouble,
  Building2,
  CalendarDays,
  Car,
  Footprints,
  Layers,
  LayoutGrid,
  MapPin,
  PawPrint,
  Plug,
  Ruler,
  ShieldCheck,
  Sofa,
  Sparkles,
  Sun,
  TrainFront,
  WashingMachine,
} from "lucide-react";
import type { FloorPlan, Property } from "../types";
import { getProperties } from "../api";
import { CostOfTenancy } from "./CostOfTenancy";
import { PropertyScreening } from "./PropertyScreening";
import { PropGallery } from "./PropPhoto";
import {
  LAUNDRY_LABELS,
  PARKING_LABELS,
  bedRangeLabel,
  bedShort,
  isNum,
  money,
  shortDate,
  sortedPlans,
} from "./PropertyDetail";

/** "3" or "2.5" — no trailing ".0" on whole numbers. */
function fmtNum(n: number): string {
  return Number.isInteger(n) ? String(n) : n.toFixed(1);
}

/** "1 – 2 bath" across all floor plans, or "1 bath" when they match. */
function bathRange(plans: FloorPlan[]): string {
  const vals = plans.map((fp) => fp.bathrooms).filter(isNum);
  if (!vals.length) return "";
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  return min === max
    ? `${fmtNum(min)} bath`
    : `${fmtNum(min)} – ${fmtNum(max)} bath`;
}

/** "520 – 1,400 sq ft" across all floor plans. */
function sqftRange(plans: FloorPlan[]): string {
  const vals = plans.map((fp) => fp.square_feet).filter(isNum);
  if (!vals.length) return "";
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  return min === max
    ? `${min.toLocaleString()} sq ft`
    : `${min.toLocaleString()} – ${max.toLocaleString()} sq ft`;
}

function cap(s: unknown): string | undefined {
  if (typeof s !== "string" || !s) return undefined;
  return s.charAt(0).toUpperCase() + s.slice(1);
}

interface Row {
  k: string;
  v: string;
}

/** Full-width section card. Renders nothing when there is nothing to show. */
function SectionCard({
  title,
  icon,
  rows,
  children,
}: {
  title: string;
  icon: ReactNode;
  rows?: Row[];
  children?: ReactNode;
}) {
  const hasRows = (rows?.length ?? 0) > 0;
  if (!hasRows && !children) return null;
  return (
    <div className="card">
      <h2 className="icon-line">
        {icon}
        {title}
      </h2>
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

function costRows(p: Property): Row[] {
  const out: Row[] = [{ k: "Monthly rent", v: `${money(p.monthly_rent)}/mo` }];
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

function requirementRows(p: Property): Row[] {
  const out: Row[] = [];
  if (isNum(p.min_credit_score))
    out.push({ k: "Credit score", v: `${p.min_credit_score} or higher` });
  if (isNum(p.min_income_multiplier))
    out.push({
      k: "Income",
      v: `Must be ${fmtNum(p.min_income_multiplier)}× the rent`,
    });
  if (isNum(p.max_occupants))
    out.push({
      k: "Household",
      v: `Up to ${p.max_occupants} ${p.max_occupants === 1 ? "person" : "people"}`,
    });
  if (typeof p.smoking_allowed === "boolean")
    out.push({ k: "Smoking", v: p.smoking_allowed ? "Allowed" : "Not allowed" });
  out.push({ k: "Pets", v: p.pets_allowed ? "Allowed" : "Not allowed" });
  if (p.pets_allowed && isNum(p.pets_max_count) && p.pets_max_count > 0)
    out.push({
      k: "Pet limit",
      v: `Up to ${p.pets_max_count} ${p.pets_max_count === 1 ? "pet" : "pets"}`,
    });
  if (p.pets_allowed && isNum(p.pet_weight_limit_lbs))
    out.push({ k: "Pet weight", v: `Up to ${p.pet_weight_limit_lbs} lbs` });
  return out;
}

function homeRows(p: Property): Row[] {
  const out: Row[] = [];
  const flooring = cap(p.flooring);
  if (flooring) out.push({ k: "Flooring", v: flooring });
  const heating = cap(p.heating);
  if (heating)
    out.push({ k: "Heating", v: heating === "None" ? "None" : `${heating} heat` });
  const cooling = cap(p.cooling);
  if (cooling)
    out.push({ k: "Cooling", v: cooling === "None" ? "None" : `${cooling} AC` });
  const laundry = p.laundry_type ? LAUNDRY_LABELS[p.laundry_type] : undefined;
  if (laundry) out.push({ k: "Laundry", v: laundry });
  const parking = PARKING_LABELS[p.parking_type];
  if (parking) out.push({ k: "Parking", v: parking });
  out.push({ k: "Furnished", v: p.furnished ? "Yes" : "No" });
  out.push({ k: "Balcony", v: p.has_balcony ? "Yes" : "No" });
  if (typeof p.storage_unit_available === "boolean")
    out.push({
      k: "Storage unit",
      v: p.storage_unit_available ? "Available" : "Not available",
    });
  if (typeof p.gated === "boolean")
    out.push({ k: "Gated community", v: p.gated ? "Yes" : "No" });
  if (isNum(p.unit_count))
    out.push({ k: "Homes in building", v: String(p.unit_count) });
  return out;
}

/** One tile in the key-facts strip under the hero. */
function Fact({
  icon,
  label,
  value,
}: {
  icon: ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="stat-tile">
      <div className="label icon-line">
        {icon}
        {label}
      </div>
      <div className="value" style={{ fontSize: 18 }}>
        {value}
      </div>
    </div>
  );
}

interface PageProps {
  propertyId: string;
  onBack: () => void;
  /** Optional pre-loaded row; the page still fetches when it's absent. */
  property?: Property;
  /**
   * Opens the Apply view with this home preselected. When absent, the
   * hero's "Apply for this home" button is not rendered.
   */
  onApply?: (propertyId: string) => void;
  /**
   * Opens the Tour Scheduler locked to this home. When absent, the hero's
   * "Book a tour" button is not rendered.
   */
  onBookTour?: (propertyId: string) => void;
  /**
   * Opens the Concierge (Ask) page scoped to this home. When absent, the
   * hero's "Ask about this home" button is not rendered.
   */
  onAsk?: (propertyId: string) => void;
}

export function PropertyPage({
  propertyId,
  onBack,
  property: preloaded,
  onApply,
  onBookTour,
  onAsk,
}: PageProps) {
  const [property, setProperty] = useState<Property | null>(
    preloaded && preloaded.id === propertyId ? preloaded : null,
  );
  const [loading, setLoading] = useState(!property);
  const [error, setError] = useState("");

  useEffect(() => {
    if (property?.id === propertyId) return; // already have the row
    let cancelled = false;
    setLoading(true);
    setError("");
    getProperties()
      .then((res) => {
        if (cancelled) return;
        const found = res.properties.find((p) => p.id === propertyId) ?? null;
        setProperty(found);
        if (!found)
          setError("We couldn't find that home. It may no longer be listed.");
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        setError(
          e instanceof TypeError
            ? "Could not reach the server. Is the backend running?"
            : e instanceof Error
              ? e.message
              : String(e),
        );
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [propertyId]);

  return (
    <div className="app">
      <button className="back-link" onClick={onBack}>
        <ArrowLeft size={14} /> Back to properties
      </button>

      {loading && (
        <p className="muted" style={{ marginTop: 18 }}>
          Loading this home…
        </p>
      )}
      {!loading && error && (
        <div className="card">
          <div className="error" role="alert" style={{ marginTop: 0 }}>
            {error}
          </div>
        </div>
      )}
      {!loading && !error && property && (
        <PageBody
          p={property}
          onApply={onApply}
          onBookTour={onBookTour}
          onAsk={onAsk}
        />
      )}
    </div>
  );
}

/** One row in the "Floor plans" section, cheapest plan first. */
function PlanRow({ fp, index }: { fp: FloorPlan; index: number }) {
  const avail = shortDate(fp.availability_date);
  const units = isNum(fp.available_units) ? fp.available_units : null;
  return (
    <div className="plan-row">
      <div className="plan-row-top">
        <span className="plan-name">{fp.name || `Plan ${index + 1}`}</span>
        <span className="icon-line">
          <BedDouble size={13} /> {bedShort(fp.bedrooms)}
        </span>
        {isNum(fp.bathrooms) && (
          <span className="icon-line">
            <Bath size={13} /> {fmtNum(fp.bathrooms)} bath
          </span>
        )}
        <span className="icon-line">
          <Ruler size={13} /> {fp.square_feet.toLocaleString()} sq ft
        </span>
        <span className="plan-rent">
          {money(fp.monthly_rent)}
          <span>/mo</span>
        </span>
      </div>
      <div className="plan-row-meta">
        {units != null && (
          <span
            className={`badge icon-line ${units > 0 ? "tone-good" : "tone-bad"}`}
          >
            {units === 1 ? "1 unit available" : `${units} units available`}
          </span>
        )}
        {avail && (
          <span className="icon-line muted">
            <CalendarDays size={13} /> Available {avail}
          </span>
        )}
      </div>
    </div>
  );
}

function PageBody({
  p,
  onApply,
  onBookTour,
  onAsk,
}: {
  p: Property;
  onApply?: (propertyId: string) => void;
  onBookTour?: (propertyId: string) => void;
  onAsk?: (propertyId: string) => void;
}) {
  const plans = sortedPlans(p);
  const multiPlan = plans.length > 1;
  const avail = shortDate(p.availability_date);
  const baths = multiPlan
    ? bathRange(plans)
    : `${fmtNum(p.bathrooms)} bath${
        p.bathroom_type === "shower_only" ? " (shower only)" : ""
      }`;
  const beds = multiPlan ? bedRangeLabel(plans) : `${p.bedrooms} bed`;
  const size = multiPlan ? sqftRange(plans) : `${p.square_feet.toLocaleString()} sq ft`;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: "easeOut" }}
    >
      <div style={{ marginBottom: 16 }}>
        <PropGallery
          photos={p.photo_urls?.length ? p.photo_urls : [p.photo_url]}
          alt={p.name}
        />
      </div>
      {/* Hero */}
      <div className="card">
        <div className="prop-hero">
          <div>
            <span className="badge tone-info">{p.property_type}</span>
            <h1 className="prop-hero-name">{p.name}</h1>
            <div className="icon-line muted" style={{ fontSize: 13 }}>
              <MapPin size={14} />
              {p.city ? `${p.area} · ${p.city}` : p.area}
            </div>
          </div>
          <div className="prop-hero-right">
            <div className="prop-hero-rent">
              {multiPlan && <span>From </span>}
              {money(p.monthly_rent)}
              <span> / month</span>
            </div>
            {avail && (
              <div className="icon-line muted" style={{ fontSize: 13 }}>
                <CalendarDays size={13} /> Available {avail}
              </div>
            )}
            {isNum(p.year_built) && (
              <div className="icon-line muted" style={{ fontSize: 13 }}>
                <Building2 size={13} /> Built {p.year_built}
              </div>
            )}
            {(onApply || onBookTour || onAsk) && (
              <div
                style={{
                  display: "flex",
                  gap: 8,
                  marginTop: 8,
                  flexWrap: "wrap",
                }}
              >
                {onApply && (
                  <button onClick={() => onApply(p.id)}>
                    Apply for this home
                  </button>
                )}
                {onBookTour && (
                  <button
                    className="btn-ghost"
                    onClick={() => onBookTour(p.id)}
                  >
                    Book a tour
                  </button>
                )}
                {onAsk && (
                  <button className="btn-ghost" onClick={() => onAsk(p.id)}>
                    Ask about this home
                  </button>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Key facts strip — ranges across floor plans for multi-plan buildings */}
        <div className="prop-facts">
          <Fact icon={<BedDouble size={13} />} label="Bedrooms" value={beds} />
          <Fact icon={<Bath size={13} />} label="Bathrooms" value={baths} />
          <Fact icon={<Ruler size={13} />} label="Size" value={size} />
          {!multiPlan && isNum(p.floor_level) && (
            <Fact
              icon={<Layers size={13} />}
              label="Floor"
              value={`Level ${p.floor_level}`}
            />
          )}
          <Fact
            icon={<CalendarDays size={13} />}
            label="Lease"
            value={`${p.lease_term_months} months`}
          />
        </div>

        {/* Quick highlight badges */}
        <div className="badges" style={{ marginTop: 12 }}>
          {p.pets_allowed && (
            <span className="badge tone-good icon-line">
              <PawPrint size={12} /> Pets OK
            </span>
          )}
          {p.in_unit_laundry && (
            <span className="badge tone-info icon-line">
              <WashingMachine size={12} /> In-unit laundry
            </span>
          )}
          {p.has_balcony && (
            <span className="badge tone-info icon-line">
              <Sun size={12} /> Balcony
            </span>
          )}
          {p.furnished && <span className="badge tone-info">Furnished</span>}
          {PARKING_LABELS[p.parking_type] && p.parking_type !== "none" && (
            <span className="badge tone-info icon-line">
              <Car size={12} /> {PARKING_LABELS[p.parking_type]}
            </span>
          )}
        </div>
      </div>

      {/* Every home shows its plans — single-plan homes get one row with
          unit count and availability, which the key facts don't cover. */}
      {plans.length > 0 && (
        <SectionCard
          title="Floor plans"
          icon={<LayoutGrid size={15} className="icon-muted" />}
        >
          <div className="plan-rows">
            {plans.map((fp, i) => (
              <PlanRow key={`${fp.name}-${i}`} fp={fp} index={i} />
            ))}
          </div>
        </SectionCard>
      )}

      <CostOfTenancy p={p} />

      <PropertyScreening propertyId={p.id} />

      <SectionCard
        title="Costs & fees"
        icon={<Banknote size={15} className="icon-muted" />}
        rows={costRows(p)}
      />

      <SectionCard
        title="Requirements"
        icon={<ShieldCheck size={15} className="icon-muted" />}
        rows={requirementRows(p)}
      />

      <SectionCard
        title="Home details"
        icon={<Sofa size={15} className="icon-muted" />}
        rows={homeRows(p)}
      >
        {(p.appliances?.length ?? 0) > 0 && (
          <div className="badges" style={{ marginTop: 10 }}>
            {p.appliances!.map((a) => (
              <span key={a} className="badge tone-info">
                {a}
              </span>
            ))}
          </div>
        )}
      </SectionCard>

      {(p.utilities_included?.length ?? 0) > 0 && (
        <SectionCard
          title="Utilities included"
          icon={<Plug size={15} className="icon-muted" />}
        >
          <div className="badges">
            {p.utilities_included!.map((u) => (
              <span key={u} className="badge tone-good">
                {u}
              </span>
            ))}
          </div>
        </SectionCard>
      )}

      {p.amenities.length > 0 && (
        <SectionCard
          title="Amenities"
          icon={<Sparkles size={15} className="icon-muted" />}
        >
          <div className="badges">
            {p.amenities.map((a) => (
              <span key={a} className="badge tone-info">
                {a}
              </span>
            ))}
          </div>
        </SectionCard>
      )}

      {(p.walk_score != null || p.transit_score != null) && (
        <SectionCard
          title="Location"
          icon={<MapPin size={15} className="icon-muted" />}
        >
          <div style={{ display: "flex", gap: 18, flexWrap: "wrap", fontSize: 13 }}>
            {p.walk_score != null && (
              <span className="icon-line muted">
                <Footprints size={14} /> Walk score {p.walk_score} / 100
              </span>
            )}
            {p.transit_score != null && (
              <span className="icon-line muted">
                <TrainFront size={14} /> Transit score {p.transit_score} / 100
              </span>
            )}
          </div>
        </SectionCard>
      )}

      <p className="muted" style={{ marginTop: 14, fontSize: 12 }}>
        Listing {p.id}
      </p>
    </motion.div>
  );
}
