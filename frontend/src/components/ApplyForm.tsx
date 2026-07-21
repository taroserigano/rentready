import { useMemo, useRef, useState } from "react";
import { CheckCircle2, Wand2 } from "lucide-react";
import type {
  ApplicantProfile,
  EligibilityResult,
  RecommendResponse,
  UploadResponse,
} from "../types";
import { ApplicationPdf } from "./ApplicationPdf";
import { StrengthCard } from "./StrengthCard";
import { EligibilityCard } from "./EligibilityCard";
import { Recommendations } from "./Recommendations";
import { applyForm, getEligibility, getRecommendations } from "../api";

/** Amenities offered by the sample inventory — one-tap add to the field. */
const COMMON_AMENITIES = [
  "Gym",
  "Pool",
  "Rooftop Deck",
  "Concierge",
  "Bike Storage",
  "Dog Park",
  "Playground",
  "Yoga Studio",
];

/**
 * Live income-vs-rent preview under the Budget fields. Uses the same 3×
 * rule the backend eligibility applies, so the applicant sees the likely
 * verdict before submitting — deterministic, no server round-trip.
 */
function AffordabilityHint({ income, rent }: { income: string; rent: string }) {
  const i = Number(income);
  const r = Number(rent);
  if (!(i > 0) || !(r > 0)) return null;
  const ratio = i / r;
  const burden = Math.round((r / i) * 100);
  const tone = ratio >= 3 ? "good" : ratio >= 2.5 ? "warn" : "bad";
  return (
    <div style={{ gridColumn: "1 / -1", marginTop: -4 }}>
      <span className={`badge tone-${tone}`}>
        Income is {ratio.toFixed(1)}× rent · rent is {burden}% of income ·{" "}
        {ratio >= 3 ? "meets the 3× minimum" : "below the 3× minimum"}
      </span>
    </div>
  );
}

const AREAS = [
  "Downtown",
  "East Austin",
  "Mueller",
  "North Austin",
  "South Congress",
  "Zilker",
];

const EMPLOYMENT = [
  "employed",
  "self-employed",
  "student",
  "retired",
  "unemployed",
];

const EMPLOYMENT_LABELS: Record<string, string> = {
  employed: "Employed",
  "self-employed": "Self-employed",
  student: "Student",
  retired: "Retired",
  unemployed: "Unemployed",
};

interface FormState {
  name: string;
  employment_status: string;
  monthly_income: string;
  desired_rent: string;
  credit_score: string;
  bedrooms_wanted: string;
  bathrooms_wanted: string;
  bath_type_wanted: string;
  min_square_feet: string;
  has_pets: boolean;
  needs_balcony: boolean;
  needs_parking: boolean;
  needs_in_unit_laundry: boolean;
  furnished_wanted: boolean;
  amenities: string;
  preferred_area: string;
  lease_term_wanted: string;
  // Employment
  employer: string;
  job_title: string;
  employment_length_months: string;
  other_income_monthly: string;
  // Financial
  savings_balance: string;
  monthly_debt_payments: string;
  // Rental history
  current_address: string;
  current_rent: string;
  years_at_current_address: string;
  reason_for_moving: string;
  landlord_reference: boolean;
  evictions_count: string;
  late_payments_12mo: string;
  bankruptcies_count: string;
  criminal_record: boolean;
  // Household
  household_size: string;
  co_applicants: string;
  dependents: string;
  pet_count: string;
  pet_types: string;
  vehicles_count: string;
  smoker: boolean;
  is_student: boolean;
  // Move-in
  desired_move_in: string;
  guarantor_available: boolean;
  references_count: string;
}

const INITIAL: FormState = {
  name: "",
  employment_status: "employed",
  monthly_income: "",
  desired_rent: "",
  credit_score: "",
  bedrooms_wanted: "",
  bathrooms_wanted: "",
  bath_type_wanted: "any",
  min_square_feet: "",
  has_pets: false,
  needs_balcony: false,
  needs_parking: false,
  needs_in_unit_laundry: false,
  furnished_wanted: false,
  amenities: "",
  preferred_area: "",
  lease_term_wanted: "",
  employer: "",
  job_title: "",
  employment_length_months: "",
  other_income_monthly: "",
  savings_balance: "",
  monthly_debt_payments: "",
  current_address: "",
  current_rent: "",
  years_at_current_address: "",
  reason_for_moving: "",
  landlord_reference: false,
  evictions_count: "",
  late_payments_12mo: "",
  bankruptcies_count: "",
  criminal_record: false,
  household_size: "",
  co_applicants: "",
  dependents: "",
  pet_count: "",
  pet_types: "",
  vehicles_count: "",
  smoker: false,
  is_student: false,
  desired_move_in: "",
  guarantor_available: false,
  references_count: "",
};

/** Realistic, fully-filled example — powers the "Prefill sample" button. */
const SAMPLE: FormState = {
  ...INITIAL,
  name: "Jordan Rivera",
  employment_status: "employed",
  monthly_income: "6800",
  desired_rent: "1950",
  credit_score: "728",
  bedrooms_wanted: "2",
  bathrooms_wanted: "2",
  bath_type_wanted: "full",
  min_square_feet: "850",
  needs_balcony: true,
  needs_parking: true,
  needs_in_unit_laundry: true,
  amenities: "Gym, Pool",
  preferred_area: "East Austin",
  lease_term_wanted: "12",
  employer: "Northstar Health",
  job_title: "Nurse Practitioner",
  employment_length_months: "36",
  other_income_monthly: "400",
  savings_balance: "18000",
  monthly_debt_payments: "350",
  current_address: "88 Riverside Dr, Austin, TX",
  current_rent: "1650",
  years_at_current_address: "3",
  reason_for_moving: "Closer to work",
  landlord_reference: true,
  household_size: "2",
  desired_move_in: "2026-09-01",
  references_count: "2",
};

type Errors = Partial<Record<keyof FormState, string>>;

function optionalNumber(value: string): number | null {
  return value.trim() === "" ? null : Number(value);
}

// Blank means "use the default" (e.g. 0 evictions, household of 1).
function numberOr(value: string, fallback: number): number {
  return value.trim() === "" ? fallback : Number(value);
}

export function ApplyForm() {
  const [form, setForm] = useState<FormState>(INITIAL);
  const [errors, setErrors] = useState<Errors>({});
  const [serverError, setServerError] = useState("");
  const [loading, setLoading] = useState(false);
  const [upload, setUpload] = useState<UploadResponse | null>(null);
  const [eligibility, setEligibility] = useState<EligibilityResult | null>(
    null,
  );
  const [recs, setRecs] = useState<RecommendResponse | null>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  const selectedAmenities = useMemo(
    () => form.amenities.split(",").map((s) => s.trim()).filter(Boolean),
    [form.amenities],
  );

  function setField<K extends keyof FormState>(key: K, value: FormState[K]) {
    setForm((f) => ({ ...f, [key]: value }));
    // Clear this field's error as soon as the user edits it.
    setErrors((e) => (e[key] ? { ...e, [key]: undefined } : e));
  }

  function toggleAmenity(a: string) {
    const set = new Set(selectedAmenities);
    if (set.has(a)) set.delete(a);
    else set.add(a);
    setField("amenities", [...set].join(", "));
  }

  function validate(): Errors {
    const next: Errors = {};
    if (!form.name.trim()) {
      next.name = "Please enter your name";
    }
    if (!(Number(form.monthly_income) > 0)) {
      next.monthly_income = "Monthly income must be more than 0";
    }
    if (!(Number(form.desired_rent) > 0)) {
      next.desired_rent = "Desired rent must be more than 0";
    }
    if (form.credit_score.trim() !== "") {
      const score = Number(form.credit_score);
      if (!(score >= 300 && score <= 850)) {
        next.credit_score = "Credit score must be between 300 and 850";
      }
    }
    return next;
  }

  function startOver() {
    setForm(INITIAL);
    setErrors({});
    setServerError("");
    setUpload(null);
    setEligibility(null);
    setRecs(null);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const nextErrors = validate();
    setErrors(nextErrors);
    if (Object.values(nextErrors).some(Boolean)) return;

    const profile: ApplicantProfile = {
      name: form.name.trim(),
      monthly_income: Number(form.monthly_income),
      desired_rent: Number(form.desired_rent),
      credit_score: optionalNumber(form.credit_score),
      employment_status: form.employment_status,
      bedrooms_wanted: optionalNumber(form.bedrooms_wanted),
      bathrooms_wanted: optionalNumber(form.bathrooms_wanted),
      bath_type_wanted: form.bath_type_wanted,
      min_square_feet: optionalNumber(form.min_square_feet),
      has_pets: form.has_pets,
      needs_balcony: form.needs_balcony,
      needs_parking: form.needs_parking,
      needs_in_unit_laundry: form.needs_in_unit_laundry,
      furnished_wanted: form.furnished_wanted,
      preferred_area: form.preferred_area,
      wanted_amenities: form.amenities
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
      lease_term_wanted: optionalNumber(form.lease_term_wanted),
      // Employment
      employer: form.employer.trim(),
      job_title: form.job_title.trim(),
      employment_length_months: optionalNumber(form.employment_length_months),
      other_income_monthly: numberOr(form.other_income_monthly, 0),
      // Financial
      savings_balance: optionalNumber(form.savings_balance),
      monthly_debt_payments: numberOr(form.monthly_debt_payments, 0),
      // Rental history
      current_address: form.current_address.trim(),
      current_rent: optionalNumber(form.current_rent),
      years_at_current_address: optionalNumber(form.years_at_current_address),
      reason_for_moving: form.reason_for_moving.trim(),
      landlord_reference: form.landlord_reference,
      evictions_count: numberOr(form.evictions_count, 0),
      late_payments_12mo: numberOr(form.late_payments_12mo, 0),
      bankruptcies_count: numberOr(form.bankruptcies_count, 0),
      criminal_record: form.criminal_record,
      // Household
      household_size: numberOr(form.household_size, 1),
      co_applicants: numberOr(form.co_applicants, 0),
      dependents: numberOr(form.dependents, 0),
      pet_count: numberOr(form.pet_count, 0),
      pet_types: form.pet_types
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
      vehicles_count: numberOr(form.vehicles_count, 0),
      smoker: form.smoker,
      is_student: form.is_student,
      // Move-in
      desired_move_in: form.desired_move_in,
      guarantor_available: form.guarantor_available,
      references_count: numberOr(form.references_count, 0),
    };

    setServerError("");
    setLoading(true);
    setUpload(null);
    setEligibility(null);
    setRecs(null);
    try {
      const created: UploadResponse = await applyForm(profile);
      const [elig, rec] = await Promise.all([
        getEligibility(created.applicant_id),
        getRecommendations(created.applicant_id),
      ]);
      setUpload(created);
      setEligibility(elig);
      setRecs(rec);
      // Let the results paint, then bring them into view.
      requestAnimationFrame(() =>
        resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }),
      );
    } catch (e) {
      // fetch throws TypeError when the server is unreachable; anything
      // else carries the server's plain-English detail message.
      setServerError(
        e instanceof TypeError
          ? "Could not reach the server. Is the backend running?"
          : e instanceof Error && e.message
            ? e.message
            : "Something went wrong. Please try again.",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Apply as a renter</h1>
        <p>Fill in your details — no PDF needed.</p>
      </header>

      <div className="card">
        <div className="rec-head" style={{ marginBottom: 8 }}>
          <h2 style={{ margin: 0 }}>Your details</h2>
          <button
            type="button"
            className="btn-small btn-ghost icon-line"
            style={{ marginLeft: "auto" }}
            onClick={() => {
              setForm(SAMPLE);
              setErrors({});
            }}
          >
            <Wand2 size={14} /> Prefill sample
          </button>
        </div>
        <form onSubmit={handleSubmit}>
          <div className="form-grid">
            <div className="form-section">About you</div>
            <Field label="Name" error={errors.name}>
              <input
                type="text"
                value={form.name}
                className={errors.name ? "invalid" : ""}
                onChange={(e) => setField("name", e.target.value)}
                placeholder="Jane Doe"
              />
            </Field>
            <Field label="Employment status">
              <select
                value={form.employment_status}
                onChange={(e) => setField("employment_status", e.target.value)}
              >
                {EMPLOYMENT.map((v) => (
                  <option key={v} value={v}>
                    {EMPLOYMENT_LABELS[v]}
                  </option>
                ))}
              </select>
            </Field>

            <div className="form-section">Budget</div>
            <Field label="Monthly income $" error={errors.monthly_income}>
              <input
                type="number"
                min={0}
                value={form.monthly_income}
                className={errors.monthly_income ? "invalid" : ""}
                onChange={(e) => setField("monthly_income", e.target.value)}
                placeholder="5000"
              />
            </Field>
            <Field label="Desired rent $" error={errors.desired_rent}>
              <input
                type="number"
                min={0}
                value={form.desired_rent}
                className={errors.desired_rent ? "invalid" : ""}
                onChange={(e) => setField("desired_rent", e.target.value)}
                placeholder="1500"
              />
            </Field>
            <Field label="Credit score (optional)" error={errors.credit_score}>
              <input
                type="number"
                min={300}
                max={850}
                value={form.credit_score}
                className={errors.credit_score ? "invalid" : ""}
                onChange={(e) => setField("credit_score", e.target.value)}
                placeholder="700"
              />
            </Field>
            <AffordabilityHint
              income={form.monthly_income}
              rent={form.desired_rent}
            />

            <div className="form-section">Unit preferences</div>
            <Field label="Bedrooms wanted">
              <input
                type="number"
                min={0}
                max={5}
                value={form.bedrooms_wanted}
                onChange={(e) => setField("bedrooms_wanted", e.target.value)}
                placeholder="2"
              />
            </Field>
            <Field label="Bathrooms wanted">
              <input
                type="number"
                min={0}
                step={0.5}
                value={form.bathrooms_wanted}
                onChange={(e) => setField("bathrooms_wanted", e.target.value)}
                placeholder="1.5"
              />
            </Field>
            <Field label="Bathroom type">
              <select
                value={form.bath_type_wanted}
                onChange={(e) => setField("bath_type_wanted", e.target.value)}
              >
                <option value="any">Any</option>
                <option value="full">Full bath</option>
                <option value="shower_only">Shower only</option>
              </select>
            </Field>
            <Field label="Minimum square feet (optional)">
              <input
                type="number"
                min={0}
                value={form.min_square_feet}
                onChange={(e) => setField("min_square_feet", e.target.value)}
                placeholder="650"
              />
            </Field>

            <div className="form-section">Lifestyle</div>
            <div className="check-row">
              <label className="check">
                <input
                  type="checkbox"
                  checked={form.has_pets}
                  onChange={(e) => setField("has_pets", e.target.checked)}
                />
                I have pets
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  checked={form.needs_balcony}
                  onChange={(e) => setField("needs_balcony", e.target.checked)}
                />
                Need balcony
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  checked={form.needs_parking}
                  onChange={(e) => setField("needs_parking", e.target.checked)}
                />
                Need parking
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  checked={form.needs_in_unit_laundry}
                  onChange={(e) =>
                    setField("needs_in_unit_laundry", e.target.checked)
                  }
                />
                Need in-unit laundry
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  checked={form.furnished_wanted}
                  onChange={(e) =>
                    setField("furnished_wanted", e.target.checked)
                  }
                />
                Want furnished
              </label>
            </div>
            <label className="field" style={{ gridColumn: "1 / -1" }}>
              <span>Amenities</span>
              <input
                type="text"
                value={form.amenities}
                onChange={(e) => setField("amenities", e.target.value)}
                placeholder="Gym, Pool"
              />
              <span className="field-hint">
                Separate with commas, like: Gym, Pool
              </span>
              <div className="chip-row">
                {COMMON_AMENITIES.map((a) => {
                  const on = selectedAmenities.includes(a);
                  return (
                    <button
                      type="button"
                      key={a}
                      className="chip"
                      aria-pressed={on}
                      style={
                        on
                          ? {
                              borderColor: "var(--accent)",
                              color: "var(--accent-text)",
                              background: "var(--accent-soft)",
                            }
                          : undefined
                      }
                      onClick={() => toggleAmenity(a)}
                    >
                      {on ? "✓ " : "+ "}
                      {a}
                    </button>
                  );
                })}
              </div>
            </label>

            <div className="form-section">Location &amp; lease</div>
            <Field label="Preferred area">
              <select
                value={form.preferred_area}
                onChange={(e) => setField("preferred_area", e.target.value)}
              >
                <option value="">No preference</option>
                {AREAS.map((a) => (
                  <option key={a} value={a}>
                    {a}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Lease term in months (optional)">
              <input
                type="number"
                min={1}
                value={form.lease_term_wanted}
                onChange={(e) => setField("lease_term_wanted", e.target.value)}
                placeholder="12"
              />
            </Field>
          </div>

          <p className="muted form-optional-note">
            The sections below are optional. Filling them in helps us review
            your application faster.
          </p>

          <details className="form-details">
            <summary>Employment</summary>
            <div className="form-grid">
              <Field label="Employer">
                <input
                  type="text"
                  value={form.employer}
                  onChange={(e) => setField("employer", e.target.value)}
                  placeholder="Acme Corp"
                />
              </Field>
              <Field label="Job title">
                <input
                  type="text"
                  value={form.job_title}
                  onChange={(e) => setField("job_title", e.target.value)}
                  placeholder="Software Engineer"
                />
              </Field>
              <Field label="Months at this job">
                <input
                  type="number"
                  min={0}
                  value={form.employment_length_months}
                  onChange={(e) =>
                    setField("employment_length_months", e.target.value)
                  }
                  placeholder="24"
                />
              </Field>
              <Field label="Other monthly income $">
                <input
                  type="number"
                  min={0}
                  value={form.other_income_monthly}
                  onChange={(e) =>
                    setField("other_income_monthly", e.target.value)
                  }
                  placeholder="0"
                />
              </Field>
            </div>
          </details>

          <details className="form-details">
            <summary>Financial</summary>
            <div className="form-grid">
              <Field label="Savings balance $">
                <input
                  type="number"
                  min={0}
                  value={form.savings_balance}
                  onChange={(e) => setField("savings_balance", e.target.value)}
                  placeholder="10000"
                />
              </Field>
              <Field label="Monthly debt payments $">
                <input
                  type="number"
                  min={0}
                  value={form.monthly_debt_payments}
                  onChange={(e) =>
                    setField("monthly_debt_payments", e.target.value)
                  }
                  placeholder="0"
                />
              </Field>
            </div>
          </details>

          <details className="form-details">
            <summary>Rental history</summary>
            <div className="form-grid">
              <label className="field" style={{ gridColumn: "1 / -1" }}>
                <span>Current address</span>
                <input
                  type="text"
                  value={form.current_address}
                  onChange={(e) => setField("current_address", e.target.value)}
                  placeholder="123 Main St, Austin, TX"
                />
              </label>
              <Field label="Current rent $">
                <input
                  type="number"
                  min={0}
                  value={form.current_rent}
                  onChange={(e) => setField("current_rent", e.target.value)}
                  placeholder="1400"
                />
              </Field>
              <Field label="Years at current address">
                <input
                  type="number"
                  min={0}
                  step={0.5}
                  value={form.years_at_current_address}
                  onChange={(e) =>
                    setField("years_at_current_address", e.target.value)
                  }
                  placeholder="2"
                />
              </Field>
              <label className="field" style={{ gridColumn: "1 / -1" }}>
                <span>Reason for moving</span>
                <input
                  type="text"
                  value={form.reason_for_moving}
                  onChange={(e) =>
                    setField("reason_for_moving", e.target.value)
                  }
                  placeholder="Closer to work"
                />
              </label>
              <Field label="Evictions">
                <input
                  type="number"
                  min={0}
                  value={form.evictions_count}
                  onChange={(e) => setField("evictions_count", e.target.value)}
                  placeholder="0"
                />
              </Field>
              <Field label="Late rent payments in the last 12 months">
                <input
                  type="number"
                  min={0}
                  value={form.late_payments_12mo}
                  onChange={(e) =>
                    setField("late_payments_12mo", e.target.value)
                  }
                  placeholder="0"
                />
              </Field>
              <Field label="Bankruptcies">
                <input
                  type="number"
                  min={0}
                  value={form.bankruptcies_count}
                  onChange={(e) =>
                    setField("bankruptcies_count", e.target.value)
                  }
                  placeholder="0"
                />
              </Field>
              <div className="check-row">
                <label className="check">
                  <input
                    type="checkbox"
                    checked={form.landlord_reference}
                    onChange={(e) =>
                      setField("landlord_reference", e.target.checked)
                    }
                  />
                  My current landlord can give a reference
                </label>
                <label className="check">
                  <input
                    type="checkbox"
                    checked={form.criminal_record}
                    onChange={(e) =>
                      setField("criminal_record", e.target.checked)
                    }
                  />
                  I have a criminal record
                </label>
              </div>
            </div>
          </details>

          <details className="form-details">
            <summary>Household</summary>
            <div className="form-grid">
              <Field label="People in your household">
                <input
                  type="number"
                  min={1}
                  value={form.household_size}
                  onChange={(e) => setField("household_size", e.target.value)}
                  placeholder="1"
                />
              </Field>
              <Field label="Co-applicants (people applying with you)">
                <input
                  type="number"
                  min={0}
                  value={form.co_applicants}
                  onChange={(e) => setField("co_applicants", e.target.value)}
                  placeholder="0"
                />
              </Field>
              <Field label="Dependents (children or others you support)">
                <input
                  type="number"
                  min={0}
                  value={form.dependents}
                  onChange={(e) => setField("dependents", e.target.value)}
                  placeholder="0"
                />
              </Field>
              <Field label="Number of pets">
                <input
                  type="number"
                  min={0}
                  value={form.pet_count}
                  onChange={(e) => setField("pet_count", e.target.value)}
                  placeholder="0"
                />
              </Field>
              <label className="field" style={{ gridColumn: "1 / -1" }}>
                <span>Kinds of pets</span>
                <input
                  type="text"
                  value={form.pet_types}
                  onChange={(e) => setField("pet_types", e.target.value)}
                  placeholder="Dog, Cat"
                />
                <span className="field-hint">
                  Separate with commas, like: Dog, Cat
                </span>
              </label>
              <Field label="Vehicles">
                <input
                  type="number"
                  min={0}
                  value={form.vehicles_count}
                  onChange={(e) => setField("vehicles_count", e.target.value)}
                  placeholder="1"
                />
              </Field>
              <div className="check-row">
                <label className="check">
                  <input
                    type="checkbox"
                    checked={form.smoker}
                    onChange={(e) => setField("smoker", e.target.checked)}
                  />
                  I smoke
                </label>
                <label className="check">
                  <input
                    type="checkbox"
                    checked={form.is_student}
                    onChange={(e) => setField("is_student", e.target.checked)}
                  />
                  I am a student
                </label>
              </div>
            </div>
          </details>

          <details className="form-details">
            <summary>Move-in</summary>
            <div className="form-grid">
              <Field label="When do you want to move in?">
                <input
                  type="date"
                  value={form.desired_move_in}
                  onChange={(e) => setField("desired_move_in", e.target.value)}
                />
              </Field>
              <Field label="References (people who can vouch for you)">
                <input
                  type="number"
                  min={0}
                  value={form.references_count}
                  onChange={(e) =>
                    setField("references_count", e.target.value)
                  }
                  placeholder="0"
                />
              </Field>
              <div className="check-row">
                <label className="check">
                  <input
                    type="checkbox"
                    checked={form.guarantor_available}
                    onChange={(e) =>
                      setField("guarantor_available", e.target.checked)
                    }
                  />
                  I have a guarantor (someone who will co-sign the lease)
                </label>
              </div>
            </div>
          </details>

          <div className="form-footer">
            {serverError && <div className="error">{serverError}</div>}
            <button type="submit" disabled={loading}>
              {loading ? "Checking…" : "Check my eligibility"}
            </button>
          </div>
        </form>
      </div>

      {upload && eligibility && recs && (
        <div ref={resultsRef}>
          <div
            className="rec-head"
            style={{ marginTop: 18, marginBottom: 4 }}
          >
            <span className="badge tone-good icon-line">
              <CheckCircle2 size={13} /> Application submitted
            </span>
            <button
              type="button"
              className="btn-small btn-ghost"
              style={{ marginLeft: "auto" }}
              onClick={startOver}
            >
              Start over
            </button>
          </div>
          <EligibilityCard result={eligibility} applicantId={upload.applicant_id} />
          <StrengthCard applicantId={upload.applicant_id} />
          {upload.has_pdf && <ApplicationPdf applicantId={upload.applicant_id} />}
          <Recommendations
            data={recs}
            applicantId={upload.applicant_id}
            monthlyIncome={upload.profile.monthly_income}
          />
        </div>
      )}
    </div>
  );
}

function Field({
  label,
  error,
  children,
}: {
  label: string;
  error?: string;
  children: React.ReactNode;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
      {error && <div className="field-error">{error}</div>}
    </label>
  );
}
