"""Deterministic residential-lease text generation.

Pure functions only — NO I/O, NO network, NO LLM. Given a property dict (the
shape returned by ``graph.load_properties``), we render a standard Texas
residential lease as ~18 titled sections, injecting the property's own
structured facts so the prose is always CONSISTENT with the data
(same rent, term, deposit, pet policy, parking, utilities, amenities).

The knowledge base (``knowledge.py``) ingests these sections as retrievable
chunks; the concierge cites them by section title.

Public API:
    lease_sections(prop) -> list[tuple[str, str]]   # (title, text) per section
    lease_markdown(prop) -> str                      # the whole lease as markdown
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _money(amount) -> str:
    """Render a dollar amount without spurious cents: 2200 -> "$2,200"."""
    try:
        value = float(amount or 0)
    except (TypeError, ValueError):
        value = 0.0
    if value == int(value):
        return f"${int(value):,}"
    return f"${value:,.2f}"


def _city(prop: dict) -> str:
    return (prop.get("neighborhood") or {}).get("city") or "Austin"


def _deposit(prop: dict) -> float:
    """Security deposit: the structured field when present, else one month rent."""
    dep = prop.get("security_deposit")
    if dep is None or float(dep) <= 0:
        return float(prop.get("monthly_rent") or 0)
    return float(dep)


def _parking_phrase(prop: dict) -> str:
    ptype = (prop.get("parking_type") or "none").lower()
    fee = float(prop.get("parking_fee_monthly") or 0)
    fee_txt = (
        f" for an additional {_money(fee)} per month"
        if fee > 0
        else " at no additional charge"
    )
    names = {
        "garage": "an assigned covered garage space",
        "covered": "an assigned covered parking space",
        "surface": "an assigned surface-lot parking space",
        "street": "street parking (permit issued by the city)",
        "carport": "an assigned carport space",
        "none": "no dedicated parking",
    }
    label = names.get(ptype, f"{ptype} parking")
    if ptype in ("none", "street"):
        # No landlord-charged fee applies to these.
        return label
    return f"{label}{fee_txt}"


# ---------------------------------------------------------------------------
# Per-section renderers. Each returns the section BODY text.
# ---------------------------------------------------------------------------
def _sec_parties(p) -> str:
    return (
        f"This Residential Lease Agreement (the \"Lease\") is entered into "
        f"between the Landlord, {p['name']} Management LLC, and the Tenant. "
        f"The Landlord leases to the Tenant the residential premises known as "
        f"{p['name']} (Property ID {p['id']}), a {p.get('bedrooms', 0)}-bedroom, "
        f"{p.get('bathrooms', 0)}-bathroom {str(p.get('property_type', 'residence')).lower()} "
        f"of approximately {p.get('square_feet', 0):,} square feet, located in the "
        f"{(p.get('neighborhood') or {}).get('name', '')} neighborhood of "
        f"{_city(p)}, Texas (the \"Premises\"). The Premises are leased for "
        f"residential use only."
    )


def _sec_term(p) -> str:
    months = int(p.get("lease_term_months") or 12)
    return (
        f"The initial term of this Lease is {months} months, commencing on the "
        f"agreed move-in date and continuing for {months} consecutive months "
        f"unless terminated earlier as provided herein. The availability date "
        f"for the Premises is {p.get('availability_date', 'as posted')}. Upon "
        f"expiration of the initial term, this Lease renews as described in the "
        f"Renewal section."
    )


def _sec_rent(p) -> str:
    rent = _money(p.get("monthly_rent"))
    return (
        f"Tenant shall pay monthly rent of {rent}, due in advance on the FIRST "
        f"(1st) day of each calendar month. Rent is considered late if not "
        f"received by the end of the fifth (5th) day of the month. Payment shall "
        f"be made by the methods designated by the Landlord. Rent does not "
        f"include optional charges such as parking or pet rent, which are billed "
        f"separately as described in this Lease."
    )


def _sec_deposit(p) -> str:
    dep = _money(_deposit(p))
    return (
        f"Tenant shall pay a refundable security deposit of {dep} prior to "
        f"occupancy. The security deposit secures Tenant's performance under "
        f"this Lease and is refundable, less lawful deductions for damage beyond "
        f"normal wear and tear and any unpaid amounts, within thirty (30) days "
        f"after Tenant surrenders the Premises, as required by Texas Property "
        f"Code Chapter 92. An application fee of "
        f"{_money(p.get('application_fee'))} and an administrative fee of "
        f"{_money(p.get('admin_fee'))} are non-refundable."
    )


def _sec_occupancy(p) -> str:
    occ = p.get("max_occupants") or (int(p.get("bedrooms") or 1) * 2)
    return (
        f"The Premises may be occupied only by the Tenant(s) named on this Lease "
        f"and their minor children, up to a maximum of {occ} occupants. Guests "
        f"may stay for no more than fourteen (14) consecutive days, or thirty "
        f"(30) days total in any twelve-month period, without the Landlord's "
        f"prior written consent. Any person who occupies the Premises beyond "
        f"these limits is considered an unauthorized occupant and a breach of "
        f"this Lease."
    )


# Property types where units share walls/floors with neighbors, as opposed to
# a standalone House — used only to word the noise section's framing.
_MULTI_UNIT_TYPES = {
    "apartment", "condo", "loft", "townhome", "townhouse", "duplex", "studio",
}


def _sec_noise(p) -> str:
    ptype = str(p.get("property_type", "")).lower()
    setting = (
        "shared walls, floors, and common areas with neighboring units"
        if ptype in _MULTI_UNIT_TYPES
        else "close proximity to neighboring properties"
    )
    return (
        f"Tenant shall not create or permit any noise, disturbance, or nuisance "
        f"that unreasonably interferes with the quiet enjoyment of other "
        f"residents or neighbors, given the Premises' {setting}. QUIET HOURS are "
        f"10:00 PM to 8:00 AM daily, during which loud music, television, "
        f"amplified sound, parties, and other disruptive noise are prohibited. "
        f"Reasonable household noise at other hours is permitted, but repeated "
        f"or substantiated complaints of excessive noise, disorderly conduct, or "
        f"nuisance behavior — whether from Tenant, occupants, or guests — "
        f"constitute a breach of this Lease and may result in a written warning "
        f"followed by termination for repeated violations. Landlord may enforce "
        f"this section based on written complaints from neighbors or its agents' "
        f"own observation."
    )


def _sec_pets(p) -> str:
    if not p.get("pets_allowed"):
        return (
            "No pets or animals of any kind are permitted on the Premises. This "
            "prohibition does NOT apply to assistance animals (service animals "
            "or emotional-support animals) required as a reasonable accommodation "
            "under applicable law, for which no pet deposit or pet rent is "
            "charged. Keeping an unauthorized pet is a breach of this Lease and "
            "may result in additional charges and termination."
        )
    pet_dep = float(p.get("pet_deposit") or 0) or 300.0
    pet_rent = float(p.get("pet_rent_monthly") or 0) or 25.0
    max_pets = int(p.get("pets_max_count") or 0) or 2
    weight = p.get("pet_weight_limit_lbs")
    weight_txt = (
        f" No single pet may exceed {int(weight)} lbs."
        if weight
        else ""
    )
    return (
        f"Pets are permitted with the Landlord's prior written consent and a pet "
        f"addendum. Tenant shall pay a refundable pet deposit of "
        f"{_money(pet_dep)} and pet rent of {_money(pet_rent)} per month per "
        f"pet. A maximum of {max_pets} pet(s) is allowed.{weight_txt} Breed and "
        f"weight restrictions apply, and aggressive breeds are prohibited. "
        f"Assistance animals required as a reasonable accommodation are exempt "
        f"from pet deposit and pet rent."
    )


def _sec_parking(p) -> str:
    return (
        f"Parking is provided as {_parking_phrase(p)}. Tenant shall park only in "
        f"assigned or designated areas, keep vehicles operational and currently "
        f"registered, and not perform vehicle repairs on the Premises. "
        f"Unauthorized or improperly parked vehicles may be towed at the "
        f"owner's expense."
    )


def _sec_utilities(p) -> str:
    included = p.get("utilities_included") or []
    if included:
        inc = ", ".join(included)
        included_txt = (
            f"The following utilities are INCLUDED in the rent and paid by the "
            f"Landlord: {inc}. "
        )
    else:
        included_txt = "No utilities are included in the rent. "
    return (
        f"{included_txt}All other utilities and services — including electricity, "
        f"gas, water/sewer (unless listed above), internet, and cable — are the "
        f"Tenant's responsibility, and Tenant shall place such accounts in "
        f"Tenant's own name effective on the move-in date. Tenant shall not "
        f"allow any utility to be disconnected during the term."
    )


def _sec_maintenance(p) -> str:
    return (
        "Landlord shall maintain the Premises in a habitable condition and make "
        "necessary repairs to structural elements, plumbing, electrical, "
        "heating, and cooling systems within a reasonable time after written "
        "notice, as required by Texas Property Code Chapter 92. Tenant shall "
        "keep the Premises clean and sanitary, promptly report needed repairs, "
        "and be responsible for damage caused by Tenant's negligence or misuse. "
        "Tenant shall not tamper with smoke or carbon-monoxide detectors."
    )


def _sec_entry(p) -> str:
    return (
        "Landlord or its agents may enter the Premises to inspect, make repairs, "
        "supply services, or show the unit to prospective tenants, purchasers, "
        "or inspectors. Except in an emergency, Landlord shall provide the "
        "Tenant at least twenty-four (24) hours' advance notice and enter only "
        "at reasonable times. In an emergency, Landlord may enter without notice."
    )


def _sec_alterations(p) -> str:
    furnished_txt = (
        "The Premises are provided FURNISHED; Tenant shall return all furnishings "
        "in their original condition, normal wear and tear excepted. "
        if p.get("furnished")
        else "The Premises are provided UNFURNISHED. "
    )
    return (
        f"{furnished_txt}Tenant shall not paint, alter, or make improvements to "
        f"the Premises, install fixtures, or change the locks without the "
        f"Landlord's prior written consent. Any approved alteration becomes part "
        f"of the Premises unless the Landlord requires its removal at the end of "
        f"the term. Tenant may hang pictures with small nails but shall repair "
        f"holes upon move-out."
    )


def _sec_sublet(p) -> str:
    return (
        "Tenant shall NOT sublet the Premises, assign this Lease, or list the "
        "unit on any short-term rental platform (such as Airbnb or VRBO) without "
        "the Landlord's prior WRITTEN consent, which the Landlord may grant or "
        "withhold at its discretion. Any purported sublease or assignment made "
        "without written consent is void and constitutes a breach of this Lease. "
        "An approved subtenant or assignee does not release the original Tenant "
        "from liability under this Lease."
    )


def _sec_late_fees(p) -> str:
    rent = float(p.get("monthly_rent") or 0)
    late = _money(round(rent * 0.05, 2))
    return (
        f"If rent is not received by the end of the fifth (5th) day of the "
        f"month, Tenant shall pay a late fee equal to five percent (5%) of the "
        f"monthly rent, which is approximately {late}. A returned or dishonored "
        f"payment (NSF) incurs a {_money(35)} returned-payment fee, and after two "
        f"such events the Landlord may require certified funds. Late fees are "
        f"additional rent and do not waive the Landlord's other remedies."
    )


def _sec_renewal(p) -> str:
    return (
        "At the end of the initial term, this Lease automatically continues on a "
        "MONTH-TO-MONTH basis unless either party gives at least sixty (60) "
        "days' written notice of non-renewal before the term ends. During any "
        "month-to-month tenancy, the Landlord may adjust the rent or terms with "
        "at least thirty (30) days' written notice. Either party may end a "
        "month-to-month tenancy with sixty (60) days' written notice."
    )


def _sec_termination(p) -> str:
    return (
        "If Tenant fails to pay rent or otherwise breaches this Lease, Landlord "
        "may deliver written notice to vacate and pursue eviction and all "
        "remedies available under Texas law. Early termination by the Tenant "
        "requires sixty (60) days' written notice and payment of a reletting "
        "fee equal to one month's rent, in addition to rent owed until the unit "
        "is re-rented; statutory rights (such as military SCRA termination and "
        "family-violence protections) are preserved. Upon termination, Tenant "
        "shall surrender the Premises clean and return all keys."
    )


def _sec_amenities(p) -> str:
    amenities = list(p.get("amenities") or [])
    extras = []
    if p.get("has_balcony"):
        extras.append("a private balcony")
    if p.get("in_unit_laundry"):
        extras.append("in-unit washer and dryer")
    if p.get("storage_unit_available"):
        extras.append("optional storage units")
    if p.get("gated"):
        extras.append("gated, controlled access")
    unit_features = (
        f" The unit includes {', '.join(extras)}." if extras else ""
    )
    if amenities:
        amen_txt = (
            f"Residents have access to the following community amenities: "
            f"{', '.join(amenities)}."
        )
    else:
        amen_txt = "This community does not offer shared amenities."
    return (
        f"{amen_txt}{unit_features} Amenities are provided for the enjoyment of "
        f"residents and their guests and are subject to posted community rules "
        f"and hours. Smoking is "
        f"{'permitted only in designated areas' if p.get('smoking_allowed') else 'PROHIBITED'} "
        f"in all units and common areas. Landlord may reasonably modify amenity "
        f"rules and hours with notice."
    )


def _sec_insurance(p) -> str:
    return (
        "Tenant shall obtain and maintain renter's insurance with personal "
        "liability coverage of at least $100,000 for the duration of the Lease, "
        "naming the Landlord as an interested party, and shall provide proof of "
        "coverage upon request. Landlord's insurance does not cover the Tenant's "
        "personal property; Tenant is encouraged to insure personal belongings."
    )


def _sec_governing(p) -> str:
    return (
        f"This Lease is governed by the laws of the State of Texas, and venue "
        f"lies in {_city(p)} and the county in which the Premises are located. "
        f"If any provision of this Lease is held unenforceable, the remaining "
        f"provisions remain in full force. This Lease, together with any "
        f"addenda, is the entire agreement between the parties and supersedes "
        f"any prior understandings. This is a binding legal document."
    )


# Ordered (title, renderer) — exactly 19 sections, matching the contract.
_SECTION_SPECS = [
    ("Parties & Premises", _sec_parties),
    ("Term", _sec_term),
    ("Rent", _sec_rent),
    ("Security Deposit", _sec_deposit),
    ("Occupancy & Guests", _sec_occupancy),
    ("Noise, Quiet Hours & Nuisance", _sec_noise),
    ("Pets", _sec_pets),
    ("Parking", _sec_parking),
    ("Utilities", _sec_utilities),
    ("Maintenance & Repairs", _sec_maintenance),
    ("Landlord Entry", _sec_entry),
    ("Alterations", _sec_alterations),
    ("Subletting & Assignment", _sec_sublet),
    ("Late Fees & Returned Payments", _sec_late_fees),
    ("Renewal", _sec_renewal),
    ("Termination & Default", _sec_termination),
    ("Community Amenities & Rules", _sec_amenities),
    ("Renter's Insurance", _sec_insurance),
    ("Governing Law", _sec_governing),
]

# Public so knowledge.py can detect a stale ingested index (built when a
# section was added/removed) without reaching into a private name.
SECTION_COUNT = len(_SECTION_SPECS)


def lease_sections(prop: dict) -> list[tuple[str, str]]:
    """Return ``[(section_title, section_text), ...]`` for one property.

    Pure and deterministic: the same property dict always yields the same
    lease. Values are drawn from the property's structured fields so the prose
    never contradicts the data.
    """
    return [(title, render(prop).strip()) for title, render in _SECTION_SPECS]


def lease_markdown(prop: dict) -> str:
    """The full lease as a single markdown document (numbered sections)."""
    lines = [
        f"# Residential Lease Agreement — {prop.get('name', 'Property')}",
        "",
        f"*Property ID: {prop.get('id', '')} · {_city(prop)}, Texas*",
        "",
    ]
    for i, (title, text) in enumerate(lease_sections(prop), start=1):
        lines.append(f"## {i}. {title}")
        lines.append("")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"
