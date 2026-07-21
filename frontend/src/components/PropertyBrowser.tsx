import { useEffect, useMemo, useState, type ReactNode } from "react";
import { motion } from "framer-motion";
import {
  ArrowUpDown,
  Banknote,
  Bath,
  BedDouble,
  Building2,
  CalendarCheck,
  Car,
  ExternalLink,
  Flame,
  Footprints,
  Heart,
  LayoutGrid,
  MapPin,
  PawPrint,
  Ruler,
  Search,
  SlidersHorizontal,
  Sun,
  WashingMachine,
  X,
} from "lucide-react";
import type { PropertiesResponse, Property } from "../types";
import { getProperties } from "../api";
import { useSavedProperties } from "../useSavedProperties";
import { PropThumb } from "./PropPhoto";
import {
  PropertyDetail,
  bathRangeLabel,
  bedRangeLabel,
  isAvailableNow,
  isNum,
  money,
  pricePerSqft,
  shortDate,
  sortedPlans,
  sqftRangeLabel,
  totalAvailableUnits,
} from "./PropertyDetail";

interface Filters {
  area: string;
  maxRent: string;
  minBedrooms: string;
  petsAllowed: boolean;
  q: string;
  // Client-side filters (all 50 rows are loaded, so no API call needed).
  maxDeposit: string;
  minYearBuilt: string;
  availableNow: boolean;
  savedOnly: boolean;
  // Advanced search (revealed by the "Advanced search" toggle) — all client-side.
  propertyType: string;
  minRent: string;
  minBathrooms: string;
  minSqft: string;
  minWalkScore: string;
  amenity: string;
  parkingType: string;
  availableBy: string;
  hasBalcony: boolean;
  inUnitLaundry: boolean;
  furnished: boolean;
  gated: boolean;
}

const EMPTY_FILTERS: Filters = {
  area: "",
  maxRent: "",
  minBedrooms: "",
  petsAllowed: false,
  q: "",
  maxDeposit: "",
  minYearBuilt: "",
  availableNow: false,
  savedOnly: false,
  propertyType: "",
  minRent: "",
  minBathrooms: "",
  minSqft: "",
  minWalkScore: "",
  amenity: "",
  parkingType: "",
  availableBy: "",
  hasBalcony: false,
  inUnitLaundry: false,
  furnished: false,
  gated: false,
};

type SortKey = "price-asc" | "price-desc" | "size" | "walk" | "value";

const SORT_OPTIONS: Array<{ value: SortKey; label: string }> = [
  { value: "price-asc", label: "Price: low to high" },
  { value: "price-desc", label: "Price: high to low" },
  { value: "value", label: "Best value ($/sq ft)" },
  { value: "size", label: "Size: biggest first" },
  { value: "walk", label: "Walk score: best first" },
];

function sortProperties(list: Property[], key: SortKey): Property[] {
  const out = [...list];
  switch (key) {
    case "price-asc":
      out.sort((a, b) => a.monthly_rent - b.monthly_rent);
      break;
    case "price-desc":
      out.sort((a, b) => b.monthly_rent - a.monthly_rent);
      break;
    case "value": {
      // Ascending $/sqft; rows with unknown sqft sort to the end.
      const ppsf = (p: Property) =>
        pricePerSqft(p.monthly_rent, p.square_feet) ?? Infinity;
      out.sort((a, b) => ppsf(a) - ppsf(b));
      break;
    }
    case "size":
      out.sort((a, b) => b.square_feet - a.square_feet);
      break;
    case "walk":
      out.sort((a, b) => (b.walk_score ?? -1) - (a.walk_score ?? -1));
      break;
  }
  return out;
}

const PARKING_LABELS: Record<string, string> = {
  garage: "Garage",
  covered: "Covered parking",
  lot: "Lot parking",
  surface: "Lot parking",
  street: "Street parking",
};

function fetchProperties(f: Filters): Promise<PropertiesResponse> {
  return getProperties({
    area: f.area || undefined,
    max_rent: f.maxRent ? Number(f.maxRent) : undefined,
    min_bedrooms: f.minBedrooms ? Number(f.minBedrooms) : undefined,
    pets_allowed: f.petsAllowed || undefined,
    q: f.q.trim() || undefined,
  });
}

function hasActiveFilters(f: Filters): boolean {
  return !!(
    f.area ||
    f.maxRent ||
    f.minBedrooms ||
    f.petsAllowed ||
    f.q.trim() ||
    f.maxDeposit ||
    f.minYearBuilt ||
    f.availableNow ||
    f.savedOnly ||
    f.propertyType ||
    f.minRent ||
    f.minBathrooms ||
    f.minSqft ||
    f.minWalkScore ||
    f.amenity.trim() ||
    f.parkingType ||
    f.availableBy ||
    f.hasBalcony ||
    f.inUnitLaundry ||
    f.furnished ||
    f.gated
  );
}

/** Human-readable labels for the active filters, for the removable chips. */
function describeFilters(f: Filters): Array<{ key: keyof Filters; label: string }> {
  const out: Array<{ key: keyof Filters; label: string }> = [];
  if (f.area) out.push({ key: "area", label: `Area: ${f.area}` });
  if (f.q.trim()) out.push({ key: "q", label: `"${f.q.trim()}"` });
  if (f.maxRent) out.push({ key: "maxRent", label: `≤ $${f.maxRent}/mo` });
  if (f.minRent) out.push({ key: "minRent", label: `≥ $${f.minRent}/mo` });
  if (f.maxDeposit) out.push({ key: "maxDeposit", label: `Deposit ≤ $${f.maxDeposit}` });
  if (f.minYearBuilt) out.push({ key: "minYearBuilt", label: `Built ≥ ${f.minYearBuilt}` });
  if (f.minBedrooms) out.push({ key: "minBedrooms", label: `${f.minBedrooms}+ beds` });
  if (f.minBathrooms) out.push({ key: "minBathrooms", label: `${f.minBathrooms}+ baths` });
  if (f.minSqft) out.push({ key: "minSqft", label: `≥ ${f.minSqft} sq ft` });
  if (f.minWalkScore) out.push({ key: "minWalkScore", label: `Walk ≥ ${f.minWalkScore}` });
  if (f.propertyType) out.push({ key: "propertyType", label: f.propertyType });
  if (f.parkingType)
    out.push({ key: "parkingType", label: PARKING_LABELS[f.parkingType] ?? f.parkingType });
  if (f.amenity.trim()) out.push({ key: "amenity", label: `Amenity: ${f.amenity.trim()}` });
  if (f.availableBy) out.push({ key: "availableBy", label: `By ${f.availableBy}` });
  if (f.availableNow) out.push({ key: "availableNow", label: "Available now" });
  if (f.savedOnly) out.push({ key: "savedOnly", label: "Saved only" });
  if (f.petsAllowed) out.push({ key: "petsAllowed", label: "Pets OK" });
  if (f.hasBalcony) out.push({ key: "hasBalcony", label: "Balcony" });
  if (f.inUnitLaundry) out.push({ key: "inUnitLaundry", label: "In-unit laundry" });
  if (f.furnished) out.push({ key: "furnished", label: "Furnished" });
  if (f.gated) out.push({ key: "gated", label: "Gated" });
  return out;
}

/**
 * Apply the client-side filters. Rows missing a field stay visible —
 * we only drop a row when we can see its value and it fails the filter.
 */
function applyClientFilters(list: Property[], f: Filters): Property[] {
  const maxDeposit = f.maxDeposit ? Number(f.maxDeposit) : null;
  const minYear = f.minYearBuilt ? Number(f.minYearBuilt) : null;
  const minRent = f.minRent ? Number(f.minRent) : null;
  const minBaths = f.minBathrooms ? Number(f.minBathrooms) : null;
  const minSqft = f.minSqft ? Number(f.minSqft) : null;
  const minWalk = f.minWalkScore ? Number(f.minWalkScore) : null;
  const amenity = f.amenity.trim().toLowerCase();
  const availableBy = f.availableBy || null;

  return list.filter((p) => {
    if (
      maxDeposit != null &&
      isNum(p.security_deposit) &&
      p.security_deposit > maxDeposit
    )
      return false;
    if (minYear != null && isNum(p.year_built) && p.year_built < minYear)
      return false;
    if (f.availableNow && !isAvailableNow(p)) return false;
    if (f.propertyType && p.property_type !== f.propertyType) return false;
    if (minRent != null && p.monthly_rent < minRent) return false;
    if (minBaths != null && isNum(p.bathrooms) && p.bathrooms < minBaths)
      return false;
    if (minSqft != null && isNum(p.square_feet) && p.square_feet < minSqft)
      return false;
    if (minWalk != null && isNum(p.walk_score) && p.walk_score < minWalk)
      return false;
    if (f.parkingType && p.parking_type !== f.parkingType) return false;
    if (f.hasBalcony && !p.has_balcony) return false;
    if (f.inUnitLaundry && !p.in_unit_laundry) return false;
    if (f.furnished && !p.furnished) return false;
    if (f.gated && !p.gated) return false;
    if (
      amenity &&
      !(p.amenities ?? []).some((a) => a.toLowerCase().includes(amenity))
    )
      return false;
    if (
      availableBy &&
      p.availability_date &&
      p.availability_date > availableBy
    )
      return false;
    return true;
  });
}

export function PropertyBrowser({
  onOpenListing,
}: {
  /** Opens the dedicated full-page listing for a property id. */
  onOpenListing?: (id: string) => void;
}) {
  const [filters, setFilters] = useState<Filters>(EMPTY_FILTERS);
  const [search, setSearch] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [sortKey, setSortKey] = useState<SortKey>("price-asc");
  const { toggle: toggleSaved, isSaved, count: savedCount } = useSavedProperties();
  const [data, setData] = useState<PropertiesResponse | null>(null);
  const [inventoryTotal, setInventoryTotal] = useState<number | null>(null);
  const [selected, setSelected] = useState<Property | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [reloadKey, setReloadKey] = useState(0);

  // Debounce the name search (~300ms) before it becomes a filter.
  useEffect(() => {
    const t = setTimeout(() => {
      setFilters((f) => (f.q === search ? f : { ...f, q: search }));
    }, 300);
    return () => clearTimeout(t);
  }, [search]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");
    fetchProperties(filters)
      .then((res) => {
        if (cancelled) return;
        setData(res);
        if (!hasActiveFilters(filters)) setInventoryTotal(res.total);
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
  }, [filters, reloadKey]);

  const set = (patch: Partial<Filters>) =>
    setFilters((f) => ({ ...f, ...patch }));

  const clearFilters = () => {
    setSearch("");
    setFilters(EMPTY_FILTERS);
  };

  const removeFilter = (key: keyof Filters) => {
    if (key === "q") setSearch("");
    set({ [key]: EMPTY_FILTERS[key] } as Partial<Filters>);
  };

  const sorted = useMemo(() => {
    let rows = applyClientFilters(data?.properties ?? [], filters);
    if (filters.savedOnly) rows = rows.filter((p) => isSaved(p.id));
    return sortProperties(rows, sortKey);
  }, [data, filters, sortKey, isSaved]);

  // Facet options for the advanced dropdowns, derived from the loaded rows.
  const propertyTypes = useMemo(
    () =>
      Array.from(
        new Set((data?.properties ?? []).map((p) => p.property_type).filter(Boolean)),
      ).sort(),
    [data],
  );
  const parkingTypes = useMemo(
    () =>
      Array.from(
        new Set((data?.properties ?? []).map((p) => p.parking_type).filter(Boolean)),
      ).sort(),
    [data],
  );

  const total = sorted.length;
  const totalHomes = inventoryTotal ?? total;
  // Re-mounting the grid when the result set or order changes re-runs
  // the stagger animation without animating on unrelated re-renders.
  const gridKey = `${sortKey}|${sorted.map((p) => p.id).join(",")}`;

  return (
    <div className="app">
      <header>
        <h1>Browse homes</h1>
        <p>Every home we know about, with filters — no application needed.</p>
      </header>

      <div className="card">
        <div className="filter-bar">
          <label className="field">
            <span>Area</span>
            <select
              value={filters.area}
              onChange={(e) => set({ area: e.target.value })}
            >
              <option value="">All areas</option>
              {(data?.areas ?? []).map((a) => (
                <option key={a} value={a}>
                  {a}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Max rent ($ / month)</span>
            <input
              type="number"
              min={0}
              placeholder="Any"
              value={filters.maxRent}
              onChange={(e) => set({ maxRent: e.target.value })}
            />
          </label>
          <label className="field">
            <span>Max deposit ($)</span>
            <input
              type="number"
              min={0}
              placeholder="Any"
              value={filters.maxDeposit}
              onChange={(e) => set({ maxDeposit: e.target.value })}
            />
          </label>
          <label className="field">
            <span>Built in or after</span>
            <input
              type="number"
              min={1900}
              max={2100}
              placeholder="Any year"
              value={filters.minYearBuilt}
              onChange={(e) => set({ minYearBuilt: e.target.value })}
            />
          </label>
          <label className="field">
            <span>Bedrooms</span>
            <select
              value={filters.minBedrooms}
              onChange={(e) => set({ minBedrooms: e.target.value })}
            >
              <option value="">Any</option>
              <option value="1">1+</option>
              <option value="2">2+</option>
              <option value="3">3+</option>
            </select>
          </label>
          <label className="field">
            <span>Home name</span>
            <div className="input-icon">
              <Search size={14} />
              <input
                type="text"
                placeholder="Search by name"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
            </div>
          </label>
          <label className="field">
            <span>Sort by</span>
            <div className="input-icon">
              <ArrowUpDown size={14} />
              <select
                value={sortKey}
                onChange={(e) => setSortKey(e.target.value as SortKey)}
              >
                {SORT_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </div>
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={filters.petsAllowed}
              onChange={(e) => set({ petsAllowed: e.target.checked })}
            />
            <PawPrint size={14} className="icon-muted" />
            Pets allowed
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={filters.availableNow}
              onChange={(e) => set({ availableNow: e.target.checked })}
            />
            <CalendarCheck size={14} className="icon-muted" />
            Available now
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={filters.savedOnly}
              onChange={(e) => set({ savedOnly: e.target.checked })}
            />
            <Heart size={14} className="icon-muted" />
            Saved only
          </label>
          <button
            type="button"
            className="btn-small btn-ghost icon-line"
            aria-expanded={showAdvanced}
            onClick={() => setShowAdvanced((v) => !v)}
          >
            <SlidersHorizontal size={14} />
            {showAdvanced ? "Hide advanced" : "Advanced search"}
          </button>
          {!loading && !error && data && (
            <span className="filter-count">
              {total} of {totalHomes} homes match
              {savedCount > 0 && ` · ${savedCount} saved`}
            </span>
          )}
        </div>

        {showAdvanced && (
          <div
            className="filter-bar"
            style={{
              marginTop: 14,
              paddingTop: 14,
              borderTop: "1px solid var(--line)",
            }}
          >
            <label className="field">
              <span>Home type</span>
              <select
                value={filters.propertyType}
                onChange={(e) => set({ propertyType: e.target.value })}
              >
                <option value="">Any type</option>
                {propertyTypes.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Min rent ($ / month)</span>
              <input
                type="number"
                min={0}
                placeholder="Any"
                value={filters.minRent}
                onChange={(e) => set({ minRent: e.target.value })}
              />
            </label>
            <label className="field">
              <span>Min bathrooms</span>
              <select
                value={filters.minBathrooms}
                onChange={(e) => set({ minBathrooms: e.target.value })}
              >
                <option value="">Any</option>
                <option value="1">1+</option>
                <option value="1.5">1.5+</option>
                <option value="2">2+</option>
                <option value="2.5">2.5+</option>
                <option value="3">3+</option>
              </select>
            </label>
            <label className="field">
              <span>Min size (sq ft)</span>
              <input
                type="number"
                min={0}
                placeholder="Any"
                value={filters.minSqft}
                onChange={(e) => set({ minSqft: e.target.value })}
              />
            </label>
            <label className="field">
              <span>Min walk score</span>
              <input
                type="number"
                min={0}
                max={100}
                placeholder="Any"
                value={filters.minWalkScore}
                onChange={(e) => set({ minWalkScore: e.target.value })}
              />
            </label>
            <label className="field">
              <span>Parking</span>
              <select
                value={filters.parkingType}
                onChange={(e) => set({ parkingType: e.target.value })}
              >
                <option value="">Any parking</option>
                {parkingTypes.map((t) => (
                  <option key={t} value={t}>
                    {PARKING_LABELS[t] ?? t}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Amenity contains</span>
              <div className="input-icon">
                <Search size={14} />
                <input
                  type="text"
                  placeholder="e.g. Pool, Gym"
                  value={filters.amenity}
                  onChange={(e) => set({ amenity: e.target.value })}
                />
              </div>
            </label>
            <label className="field">
              <span>Available by</span>
              <input
                type="date"
                value={filters.availableBy}
                onChange={(e) => set({ availableBy: e.target.value })}
              />
            </label>
            <label className="check">
              <input
                type="checkbox"
                checked={filters.hasBalcony}
                onChange={(e) => set({ hasBalcony: e.target.checked })}
              />
              <Sun size={14} className="icon-muted" />
              Balcony
            </label>
            <label className="check">
              <input
                type="checkbox"
                checked={filters.inUnitLaundry}
                onChange={(e) => set({ inUnitLaundry: e.target.checked })}
              />
              <WashingMachine size={14} className="icon-muted" />
              In-unit laundry
            </label>
            <label className="check">
              <input
                type="checkbox"
                checked={filters.furnished}
                onChange={(e) => set({ furnished: e.target.checked })}
              />
              Furnished
            </label>
            <label className="check">
              <input
                type="checkbox"
                checked={filters.gated}
                onChange={(e) => set({ gated: e.target.checked })}
              />
              Gated community
            </label>
          </div>
        )}

        {loading && (
          <p className="muted" style={{ marginTop: 12 }}>
            Loading properties…
          </p>
        )}
        {!loading && error && (
          <>
            <div className="error">{error}</div>
            <button
              className="btn-small btn-ghost"
              style={{ marginTop: 10 }}
              onClick={() => setReloadKey((k) => k + 1)}
            >
              Retry
            </button>
          </>
        )}
      </div>

      {hasActiveFilters(filters) && (
        <div className="active-filters" role="list" aria-label="Active filters">
          {describeFilters(filters).map((c) => (
            <span className="chip removable" role="listitem" key={c.key}>
              {c.label}
              <button
                className="chip-x"
                aria-label={`Remove filter: ${c.label}`}
                onClick={() => removeFilter(c.key)}
              >
                <X size={12} />
              </button>
            </span>
          ))}
          <button className="chip" onClick={clearFilters}>
            Clear all
          </button>
        </div>
      )}

      {!loading && !error && data && (
        <>
          {total === 0 ? (
            <div className="card">
              <p className="muted">
                No homes match these filters. Try raising the max rent.
              </p>
              <button className="btn-ghost" onClick={clearFilters}>
                Clear filters
              </button>
            </div>
          ) : (
            <div
              key={gridKey}
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
                gap: 16,
                marginTop: 18,
              }}
            >
              {sorted.map((p, i) => (
                <PropertyCard
                  key={p.id}
                  p={p}
                  index={i}
                  saved={isSaved(p.id)}
                  onToggleSave={() => toggleSaved(p.id)}
                  onOpen={() => setSelected(p)}
                  onOpenListing={onOpenListing}
                />
              ))}
            </div>
          )}
        </>
      )}

      <PropertyDetail
        property={selected}
        onClose={() => setSelected(null)}
        onViewListing={
          onOpenListing
            ? (id) => {
                setSelected(null);
                onOpenListing(id);
              }
            : undefined
        }
      />
    </div>
  );
}

function formatBaths(p: Property): string {
  const n = Number.isInteger(p.bathrooms)
    ? String(p.bathrooms)
    : p.bathrooms.toFixed(1);
  return `${n} bath`;
}

function chips(p: Property): Array<{ label: string; icon?: ReactNode }> {
  const out: Array<{ label: string; icon?: ReactNode }> = [];
  if (p.has_balcony) out.push({ label: "Balcony", icon: <Sun size={12} /> });
  if (p.in_unit_laundry)
    out.push({ label: "In-unit laundry", icon: <WashingMachine size={12} /> });
  if (p.pets_allowed)
    out.push({ label: "Pets OK", icon: <PawPrint size={12} /> });
  if (p.furnished) out.push({ label: "Furnished" });
  const parking = PARKING_LABELS[p.parking_type];
  if (parking) out.push({ label: parking, icon: <Car size={12} /> });
  return out;
}

/**
 * 1-2 compact fact lines from the new listing fields, e.g.
 * "Built 2018 · Available Aug 1" and "Deposit $1,450 · $50 application fee".
 * Renders nothing when a field is missing (older cached rows).
 */
function CardFacts({ p }: { p: Property }) {
  const built = isNum(p.year_built) ? `Built ${p.year_built}` : null;
  const avail = shortDate(p.availability_date);
  const line1 = [built, avail ? `Available ${avail}` : null]
    .filter(Boolean)
    .join(" · ");

  const deposit = isNum(p.security_deposit)
    ? `Deposit ${money(p.security_deposit)}`
    : null;
  const appFee =
    isNum(p.application_fee) && p.application_fee > 0
      ? `${money(p.application_fee)} application fee`
      : null;
  const line2 = [deposit, appFee].filter(Boolean).join(" · ");

  if (!line1 && !line2) return null;
  return (
    <div className="muted" style={{ fontSize: 12, marginTop: 6, display: "grid", gap: 3 }}>
      {line1 && (
        <span className="icon-line">
          <Building2 size={12} /> {line1}
        </span>
      )}
      {line2 && (
        <span className="icon-line">
          <Banknote size={12} /> {line2}
        </span>
      )}
    </div>
  );
}

function PropertyCard({
  p,
  index,
  saved,
  onToggleSave,
  onOpen,
  onOpenListing,
}: {
  p: Property;
  index: number;
  saved?: boolean;
  onToggleSave?: () => void;
  onOpen: () => void;
  onOpenListing?: (id: string) => void;
}) {
  // Properties with 2+ floor plans show ranges ("Studio – 3 bd",
  // "From $1,450 / month"). Single-plan rows (or rows without
  // floor_plans yet) render exactly as before.
  const plans = sortedPlans(p);
  const multiPlan = plans.length > 1;
  const ppsf = pricePerSqft(p.monthly_rent, p.square_feet);
  const availNow = isAvailableNow(p);
  const units = totalAvailableUnits(p);
  const statusBadges: Array<{ label: string; tone: string; icon: ReactNode }> = [];
  if (availNow)
    statusBadges.push({ label: "Available now", tone: "good", icon: <CalendarCheck size={12} /> });
  if (units === 0)
    statusBadges.push({ label: "Waitlist", tone: "bad", icon: <Flame size={12} /> });
  else if (units === 1)
    statusBadges.push({ label: "Last one", tone: "bad", icon: <Flame size={12} /> });
  else if (units === 2)
    statusBadges.push({ label: "Only 2 left", tone: "warn", icon: <Flame size={12} /> });

  return (
    <motion.div
      className="card prop-card"
      style={{ marginTop: 0 }}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2, delay: Math.min(index * 0.025, 0.25) }}
      onClick={onOpen}
      onKeyDown={(e) => {
        // Ignore key presses on inner controls (e.g. the listing button).
        if (e.target !== e.currentTarget) return;
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onOpen();
        }
      }}
      role="button"
      tabIndex={0}
      aria-label={`View details for ${p.name}`}
    >
      {onToggleSave && (
        <button
          className={`prop-save${saved ? " saved" : ""}`}
          aria-label={saved ? `Saved ${p.name}` : `Save ${p.name}`}
          aria-pressed={!!saved}
          title={saved ? "Saved" : "Save"}
          onClick={(e) => {
            e.stopPropagation();
            onToggleSave();
          }}
          onKeyDown={(e) => e.stopPropagation()}
        >
          <Heart size={16} />
        </button>
      )}
      <PropThumb src={p.photo_url || p.photo_urls?.[0]} alt={p.name} />
      <div className="prop-body">
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            gap: 8,
          }}
        >
          <div style={{ fontWeight: 600, fontSize: 15, letterSpacing: "-0.01em" }}>
            {p.name}
          </div>
          {onOpenListing && (
            <button
              className="btn-ghost prop-open-btn"
              aria-label={`Open full listing for ${p.name}`}
              title="View full listing"
              onClick={(e) => {
                e.stopPropagation();
                onOpenListing(p.id);
              }}
            >
              <ExternalLink size={14} />
            </button>
          )}
        </div>
        <div className="icon-line muted" style={{ fontSize: 12 }}>
          <MapPin size={12} />
          {p.city ? `${p.area} · ${p.city}` : p.area}
        </div>
        <div className="prop-rent" style={{ margin: "6px 0" }}>
          {multiPlan && "From "}
          <strong>{money(p.monthly_rent)}</strong>/mo
          {ppsf != null && (
            <span style={{ fontVariantNumeric: "tabular-nums" }}>
              {" "}
              · ${ppsf.toFixed(2)}/sq ft
            </span>
          )}
        </div>
        {statusBadges.length > 0 && (
          <div className="badges" style={{ marginBottom: 2 }}>
            {statusBadges.map((b) => (
              <span key={b.label} className={`badge tone-${b.tone} icon-line`}>
                {b.icon}
                {b.label}
              </span>
            ))}
          </div>
        )}
        <div
          className="muted"
          style={{ fontSize: 13, display: "flex", flexWrap: "wrap", gap: 10 }}
        >
          <span className="icon-line">
            <BedDouble size={14} />{" "}
            {multiPlan ? bedRangeLabel(plans) : `${p.bedrooms} bed`}
          </span>
          <span className="icon-line">
            <Bath size={14} />{" "}
            {multiPlan ? (bathRangeLabel(plans) ?? formatBaths(p)) : formatBaths(p)}
          </span>
          <span className="icon-line">
            <Ruler size={14} />{" "}
            {multiPlan
              ? sqftRangeLabel(plans)
              : `${p.square_feet.toLocaleString()} sq ft`}
          </span>
        </div>
        {multiPlan && (
          <div className="icon-line muted" style={{ fontSize: 12, marginTop: 4 }}>
            <LayoutGrid size={12} /> {plans.length} floor plans
          </div>
        )}
        <CardFacts p={p} />
        <div className="badges" style={{ marginTop: 8 }}>
          {chips(p).map((c) => (
            <span key={c.label} className="badge tone-info icon-line">
              {c.icon}
              {c.label}
            </span>
          ))}
        </div>
        {p.amenities.length > 0 && (
          <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>
            {p.amenities.join(", ")}
          </div>
        )}
        {(p.walk_score != null || p.transit_score != null) && (
          <div className="prop-foot">
            <span className="icon-line">
              <Footprints size={12} />
              {[
                p.walk_score != null ? `Walk ${p.walk_score}` : null,
                p.transit_score != null ? `Transit ${p.transit_score}` : null,
              ]
                .filter(Boolean)
                .join(" · ")}
            </span>
          </div>
        )}
      </div>
    </motion.div>
  );
}
