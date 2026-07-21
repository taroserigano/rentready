import { Wallet } from "lucide-react";
import type { Property } from "../types";
import { isNum, money } from "./PropertyDetail";

/**
 * True move-in cash and all-in monthly cost, computed from the fee stack the
 * listing already carries (deposits, app/admin fees, pet & parking costs).
 * Sticker rent hides these — this answers "what will it actually cost?".
 * Pure client arithmetic; renders nothing if the core numbers are missing.
 */
export function CostOfTenancy({ p }: { p: Property }) {
  const rent = p.monthly_rent;
  if (!isNum(rent) || rent <= 0) return null;

  const dep = isNum(p.security_deposit) ? p.security_deposit : 0;
  const app = isNum(p.application_fee) ? p.application_fee : 0;
  const admin = isNum(p.admin_fee) ? p.admin_fee : 0;
  const petDep = p.pets_allowed && isNum(p.pet_deposit) ? p.pet_deposit : 0;
  const petRent = p.pets_allowed && isNum(p.pet_rent_monthly) ? p.pet_rent_monthly : 0;
  const parking = isNum(p.parking_fee_monthly) ? p.parking_fee_monthly : 0;

  const moveIn = rent + dep + app + admin + petDep;
  const allIn = rent + petRent + parking;

  const upfrontRaw: Array<[string, number]> = [
    ["First month's rent", rent],
    ["Security deposit", dep],
    ["Application fee", app],
    ["Admin fee", admin],
    ["Pet deposit", petDep],
  ];
  const upfront = upfrontRaw.filter(([, v]) => v > 0);

  const monthlyRaw: Array<[string, number]> = [
    ["Base rent", rent],
    ["Pet rent", petRent],
    ["Parking", parking],
  ];
  const monthly = monthlyRaw.filter(([, v]) => v > 0);

  return (
    <div className="card">
      <h2 className="icon-line">
        <Wallet size={16} /> Cost of tenancy
      </h2>
      <div className="fh-grid">
        <div className="stat-tile">
          <div className="label">Move-in cost</div>
          <div className="value">{money(moveIn)}</div>
          <div className="sub">cash to move in</div>
        </div>
        <div className="stat-tile">
          <div className="label">All-in monthly</div>
          <div className="value">{money(allIn)}</div>
          <div className="sub">rent + recurring fees</div>
        </div>
      </div>
      <div className="grid-2col">
        <div>
          <div className="muted" style={{ fontSize: 12, margin: "10px 0 4px" }}>
            Upfront
          </div>
          {upfront.map(([label, v]) => (
            <div key={label} className="detail-row">
              <span className="muted">{label}</span>
              <span style={{ fontVariantNumeric: "tabular-nums" }}>{money(v)}</span>
            </div>
          ))}
        </div>
        <div>
          <div className="muted" style={{ fontSize: 12, margin: "10px 0 4px" }}>
            Recurring / month
          </div>
          {monthly.map(([label, v]) => (
            <div key={label} className="detail-row">
              <span className="muted">{label}</span>
              <span style={{ fontVariantNumeric: "tabular-nums" }}>{money(v)}</span>
            </div>
          ))}
          {p.utilities_included && p.utilities_included.length > 0 && (
            <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
              Utilities included: {p.utilities_included.join(", ")}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
