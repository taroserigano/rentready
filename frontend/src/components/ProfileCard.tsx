import type { ApplicantProfile } from "../types";
import { Avatar } from "./Avatar";

function money(n: number): string {
  return `$${n.toLocaleString()}`;
}

export function ProfileCard({
  profile,
  chunks,
}: {
  profile: ApplicantProfile;
  chunks: number;
}) {
  const yn = (b: boolean) => (b ? "Yes" : "No");
  const fields: [string, string][] = [
    ["Name", profile.name],
    ["Monthly income", money(profile.monthly_income)],
    ["Desired rent", money(profile.desired_rent)],
    ["Credit score", profile.credit_score?.toString() ?? "—"],
    ["Employment", profile.employment_status],
    ["Preferred area", profile.preferred_area || "—"],
    ["Bedrooms", profile.bedrooms_wanted?.toString() ?? "—"],
    [
      "Bathrooms",
      profile.bathrooms_wanted
        ? `${profile.bathrooms_wanted} (${profile.bath_type_wanted})`
        : "—",
    ],
    ["Min size", profile.min_square_feet ? `${profile.min_square_feet} sqft` : "—"],
    ["Pets", yn(profile.has_pets)],
    ["Balcony", yn(profile.needs_balcony)],
    ["Parking", yn(profile.needs_parking)],
    ["In-unit laundry", yn(profile.needs_in_unit_laundry)],
    ["Furnished", yn(profile.furnished_wanted)],
    ["Lease term", profile.lease_term_wanted ? `${profile.lease_term_wanted} mo` : "—"],
    ["Wanted amenities", profile.wanted_amenities.join(", ") || "—"],
  ];

  // Extra rows: only show fields the applicant actually filled in
  // (skip defaults so old profiles don't render a wall of blanks).
  const extra: [string, string][] = [];
  const p = profile;
  if (p.employer) {
    extra.push(["Employer", p.job_title ? `${p.job_title} at ${p.employer}` : p.employer]);
  } else if (p.job_title) {
    extra.push(["Job title", p.job_title]);
  }
  if (p.employment_length_months != null) {
    extra.push(["Time at job", `${p.employment_length_months} mo`]);
  }
  if (p.other_income_monthly) {
    extra.push(["Other income", `${money(p.other_income_monthly)}/mo`]);
  }
  if (p.savings_balance != null) extra.push(["Savings", money(p.savings_balance)]);
  if (p.monthly_debt_payments) {
    extra.push(["Debt payments", `${money(p.monthly_debt_payments)}/mo`]);
  }
  if (p.current_address) extra.push(["Current address", p.current_address]);
  if (p.current_rent != null) extra.push(["Current rent", money(p.current_rent)]);
  if (p.years_at_current_address != null) {
    extra.push(["Years at address", `${p.years_at_current_address}`]);
  }
  if (p.reason_for_moving) extra.push(["Reason for moving", p.reason_for_moving]);
  if ((p.household_size ?? 1) > 1) {
    extra.push(["Household size", `${p.household_size}`]);
  }
  if (p.co_applicants) extra.push(["Co-applicants", `${p.co_applicants}`]);
  if (p.dependents) extra.push(["Dependents", `${p.dependents}`]);
  if (p.pet_count) {
    const kinds = p.pet_types?.length ? ` (${p.pet_types.join(", ")})` : "";
    extra.push(["Pets owned", `${p.pet_count}${kinds}`]);
  }
  if (p.vehicles_count) extra.push(["Vehicles", `${p.vehicles_count}`]);
  if (p.desired_move_in) extra.push(["Desired move-in", p.desired_move_in]);
  if (p.references_count) extra.push(["References", `${p.references_count}`]);

  // Badges: quick yes/no facts and history flags worth calling out.
  const badges: { label: string; tone: string }[] = [];
  if (p.evictions_count) {
    badges.push({ label: `${p.evictions_count} eviction(s)`, tone: "tone-bad" });
  }
  if (p.late_payments_12mo) {
    badges.push({
      label: `${p.late_payments_12mo} late payment(s) in 12 mo`,
      tone: "tone-warn",
    });
  }
  if (p.bankruptcies_count) {
    badges.push({ label: `${p.bankruptcies_count} bankruptcy(ies)`, tone: "tone-warn" });
  }
  if (p.criminal_record) badges.push({ label: "Criminal record", tone: "tone-warn" });
  if (p.landlord_reference) {
    badges.push({ label: "Landlord reference", tone: "tone-good" });
  }
  if (p.guarantor_available) badges.push({ label: "Has guarantor", tone: "tone-good" });
  if (p.smoker) badges.push({ label: "Smoker", tone: "tone-info" });
  if (p.is_student) badges.push({ label: "Student", tone: "tone-info" });

  return (
    <div className="card">
      <h2>2. Extracted profile</h2>
      <div className="profile-head">
        <Avatar name={profile.name} size={48} />
        <div>
          <div className="profile-head-name">{profile.name || "Unknown"}</div>
          <div className="muted" style={{ fontSize: 13 }}>
            {profile.employment_status || "—"}
            {profile.preferred_area ? ` · prefers ${profile.preferred_area}` : ""}
          </div>
        </div>
      </div>
      <div className="profile-grid">
        {fields.map(([label, value]) => (
          <div key={label}>
            <span>{label}</span>
            {value}
          </div>
        ))}
        {extra.map(([label, value]) => (
          <div key={label}>
            <span>{label}</span>
            {value}
          </div>
        ))}
      </div>
      {badges.length > 0 && (
        <div className="badges">
          {badges.map((b) => (
            <span key={b.label} className={`badge ${b.tone}`}>
              {b.label}
            </span>
          ))}
        </div>
      )}
      <p className="muted" style={{ marginTop: 12 }}>
        Indexed {chunks} chunk(s) into ChromaDB via LlamaIndex.
      </p>
    </div>
  );
}
