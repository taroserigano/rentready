"""Generate a rental-application PDF from an ApplicantProfile.

Used when an applicant is created via the Apply form (no uploaded document),
so every application in the system has a real PDF — viewable in the Workspace
and processed by the same PDF pipeline as uploads.
"""

from models import ApplicantProfile


def _money(v) -> str:
    try:
        return f"${int(round(float(v))):,}"
    except (TypeError, ValueError):
        return str(v)


def _yn(v: bool) -> str:
    return "Yes" if v else "No"


def _sections(p: ApplicantProfile) -> list:
    """(heading, [(label, value)]) — only rows with a meaningful value."""

    def rows(*pairs):
        return [(k, v) for k, v in pairs if v not in (None, "", [], 0.0)]

    bath = None
    if p.bathrooms_wanted is not None:
        bt = {"full": " (full bath)", "shower_only": " (shower only)"}.get(
            p.bath_type_wanted, ""
        )
        bath = f"{p.bathrooms_wanted}{bt}"

    lifestyle = ", ".join(
        [
            t
            for t, on in [
                ("pets", p.has_pets),
                ("balcony", p.needs_balcony),
                ("parking", p.needs_parking),
                ("in-unit laundry", p.needs_in_unit_laundry),
                ("furnished", p.furnished_wanted),
            ]
            if on
        ]
    )

    return [
        (
            "APPLICANT",
            rows(
                ("Name", p.name),
                ("Employment status", p.employment_status),
                ("Employer", p.employer),
                ("Job title", p.job_title),
            ),
        ),
        (
            "BUDGET & CREDIT",
            rows(
                ("Monthly income", _money(p.monthly_income)),
                ("Other monthly income", _money(p.other_income_monthly) if p.other_income_monthly else None),
                ("Desired rent", _money(p.desired_rent)),
                ("Credit score", p.credit_score),
                ("Monthly debt payments", _money(p.monthly_debt_payments) if p.monthly_debt_payments else None),
                ("Savings balance", _money(p.savings_balance) if p.savings_balance else None),
            ),
        ),
        (
            "UNIT PREFERENCES",
            rows(
                ("Bedrooms wanted", p.bedrooms_wanted),
                ("Bathrooms wanted", bath),
                ("Minimum square feet", p.min_square_feet),
                ("Lifestyle needs", lifestyle),
                ("Wanted amenities", ", ".join(p.wanted_amenities)),
            ),
        ),
        (
            "LOCATION & LEASE",
            rows(
                ("Preferred area", p.preferred_area),
                ("Lease term (months)", p.lease_term_wanted),
                ("Desired move-in", p.desired_move_in),
            ),
        ),
        (
            "RENTAL HISTORY",
            rows(
                ("Current address", p.current_address),
                ("Current rent", _money(p.current_rent) if p.current_rent else None),
                ("Years at current address", p.years_at_current_address),
                ("Reason for moving", p.reason_for_moving),
                ("Landlord reference", _yn(p.landlord_reference) if p.landlord_reference else None),
                ("Past evictions", p.evictions_count or None),
                ("Late payments (12 mo)", p.late_payments_12mo or None),
                ("Bankruptcies", p.bankruptcies_count or None),
            ),
        ),
        (
            "HOUSEHOLD",
            rows(
                ("Household size", p.household_size if p.household_size and p.household_size != 1 else None),
                ("Co-applicants", p.co_applicants or None),
                ("Dependents", p.dependents or None),
                ("Pets", f"{p.pet_count} ({', '.join(p.pet_types)})" if p.pet_count else None),
                ("Vehicles", p.vehicles_count or None),
                ("Guarantor available", _yn(p.guarantor_available) if p.guarantor_available else None),
            ),
        ),
    ]


def generate_application_pdf(profile: ApplicantProfile, path: str) -> None:
    """Render the profile into a clean multi-section PDF at ``path``."""
    import fitz  # PyMuPDF

    doc = fitz.open()
    page = doc.new_page()
    left, right = 72, 523
    y = 72

    def new_page():
        nonlocal page, y
        page = doc.new_page()
        y = 72

    def line(text: str, size: int, color=(0, 0, 0), gap: float = 6, font="helv"):
        nonlocal y
        if y > 760:
            new_page()
        page.insert_text((left, y), text, fontsize=size, fontname=font, color=color)
        y += size + gap

    line("RENTAL APPLICATION", 20, gap=4, font="hebo")
    line(profile.name or "Applicant", 13, color=(0.3, 0.3, 0.3), gap=14)

    for heading, rows in _sections(profile):
        if not rows:
            continue
        if y > 720:
            new_page()
        line(heading, 11, color=(0.36, 0.42, 0.82), gap=6, font="hebo")
        for label, value in rows:
            if y > 760:
                new_page()
            page.insert_text(
                (left, y), f"{label}:", fontsize=10, fontname="helv", color=(0.4, 0.4, 0.4)
            )
            # Single-line value; truncate defensively so it never runs off-page.
            text = str(value)
            page.insert_text(
                (left + 175, y), text[:70], fontsize=10, fontname="helv", color=(0.1, 0.1, 0.1)
            )
            y += 17
        y += 10

    doc.save(path)
    doc.close()
