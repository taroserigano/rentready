"""Generate several mock rental-application PDFs into data/applications/.

Each applicant is deliberately different (income, area, pets, credit) so you
get a mix of eligibility outcomes to explore.

Run:  python scripts/make_sample_pdfs.py
"""

from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "applications"

APPLICANTS = [
    {
        "slug": "jordan-rivera",
        "name": "Jordan Rivera",
        "email": "jordan.rivera@example.com",
        "phone": "(512) 555-0199",
        "employment": "Employed full-time",
        "employer": "Lumina Software Inc.",
        "income": "6,200",
        "credit": "712",
        "rent": "1,800",
        "bedrooms": "2",
        "bathrooms": "2",
        "bath_type": "full bath",
        "sqft": "900",
        "area": "South Congress",
        "pets": "Yes - one small dog",
        "balcony": "Yes",
        "parking": "Yes - covered parking required",
        "laundry": "Yes - in-unit laundry",
        "furnished": "No",
        "lease": "12",
        "amenities": "Pet Park, Gym",
        "notes": "Relocating for work, wants to move within 30 days.",
    },
    {
        "slug": "alex-chen",
        "name": "Alex Chen",
        "email": "alex.chen@example.com",
        "phone": "(512) 555-0142",
        "employment": "Employed full-time",
        "employer": "Northstar Health",
        "income": "9,000",
        "credit": "781",
        "rent": "2,200",
        "bedrooms": "2",
        "bathrooms": "2",
        "bath_type": "full bath",
        "sqft": "1100",
        "area": "Downtown",
        "pets": "No",
        "balcony": "Yes",
        "parking": "Yes - garage",
        "laundry": "Yes - in-unit laundry",
        "furnished": "No",
        "lease": "12",
        "amenities": "Pool, Gym, Rooftop Deck",
        "notes": "Looking for a modern building close to the office.",
    },
    {
        "slug": "sam-patel",
        "name": "Sam Patel",
        "email": "sam.patel@example.com",
        "phone": "(512) 555-0177",
        "employment": "Part-time",
        "employer": "Campus Bookstore",
        "income": "3,000",
        "credit": "640",
        "rent": "1,400",
        "bedrooms": "1",
        "bathrooms": "1",
        "bath_type": "shower-only is fine",
        "sqft": "500",
        "area": "East Austin",
        "pets": "Yes - one cat",
        "balcony": "Nice to have",
        "parking": "No",
        "laundry": "No",
        "furnished": "Yes - prefer furnished",
        "lease": "6",
        "amenities": "Bike Storage",
        "notes": "Graduate student; income is below three times the rent.",
    },
    {
        "slug": "maria-gonzalez",
        "name": "Maria Gonzalez",
        "email": "maria.gonzalez@example.com",
        "phone": "(512) 555-0123",
        "employment": "Self-employed",
        "employer": "Freelance Designer",
        "income": "5,000",
        "credit": "",  # no credit score on file -> needs review
        "rent": "1,500",
        "bedrooms": "2",
        "bathrooms": "2",
        "bath_type": "full bath",
        "sqft": "950",
        "area": "North Austin",
        "pets": "Yes - two dogs",
        "balcony": "No",
        "parking": "Yes - parking needed",
        "laundry": "Yes - in-unit laundry",
        "furnished": "No",
        "lease": "12",
        "amenities": "Pet Park, Playground",
        "notes": "No credit score reported; provided bank statements instead.",
    },
    {
        "slug": "taylor-brooks",
        "name": "Taylor Brooks",
        "email": "taylor.brooks@example.com",
        "phone": "(512) 555-0188",
        "employment": "Employed full-time",
        "employer": "City Transit Authority",
        "income": "4,800",
        "credit": "600",  # below the 620 minimum -> needs review
        "rent": "1,250",
        "bedrooms": "1",
        "bathrooms": "1",
        "bath_type": "full bath",
        "sqft": "600",
        "area": "North Austin",
        "pets": "No",
        "balcony": "No",
        "parking": "Yes - parking required",
        "laundry": "No",
        "furnished": "No",
        "lease": "12",
        "amenities": "Gym",
        "notes": "Solid income but credit score is just under the minimum.",
    },
]

TEMPLATE = """RENTAL APPLICATION

Name: {name}
Email: {email}
Phone: {phone}

EMPLOYMENT
Employment Status: {employment}
Employer: {employer}
Monthly Income: ${income}
Credit Score: {credit}

HOUSING PREFERENCES
Desired Rent: ${rent}
Bedrooms: {bedrooms}
Bathrooms: {bathrooms} ({bath_type})
Minimum Size: {sqft} square feet
Preferred Area: {area}
Pets: {pets}
Balcony: {balcony}
Parking: {parking}
Laundry: {laundry}
Furnished: {furnished}
Lease Term: {lease} months
Wanted Amenities: {amenities}

NOTES
{notes}
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for app in APPLICANTS:
        text = TEMPLATE.format(**app)
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=11, fontname="helv")
        out = OUT_DIR / f"{app['slug']}.pdf"
        doc.save(str(out))
        doc.close()
        print(f"Wrote {out}")
    print(f"\nGenerated {len(APPLICANTS)} application PDFs in {OUT_DIR}")


if __name__ == "__main__":
    main()
