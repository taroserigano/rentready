"""Generate data/sample_application.pdf so the app works out of the box.

Run:  python scripts/make_sample_pdf.py
"""

from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample_application.pdf"

APPLICATION_TEXT = """RENTAL APPLICATION

Name: Jordan Rivera
Email: jordan.rivera@example.com
Phone: (512) 555-0199

EMPLOYMENT
Employment Status: Employed full-time
Employer: Lumina Software Inc.
Monthly Income: $6,200
Credit Score: 712

HOUSING PREFERENCES
Desired Rent: $1,800
Bedrooms: 2
Bathrooms: 2 (full bath)
Minimum Size: 900 square feet
Preferred Area: South Congress
Pets: Yes - one small dog
Balcony: Yes
Parking: Yes - covered parking required
Laundry: Yes - in-unit laundry
Furnished: No
Lease Term: 12 months
Wanted Amenities: Pet Park, Gym

NOTES
Applicant is relocating for work and is looking to move within 30 days.
Prefers a quiet neighborhood with easy access to downtown Austin.
"""


def main() -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), APPLICATION_TEXT, fontsize=11, fontname="helv")
    doc.save(str(OUT))
    doc.close()
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
