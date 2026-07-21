"""Manual applicant intake (the Apply form).

Lets a user create an applicant by filling a form instead of uploading a
PDF. Validation is the ApplicantProfile model itself, so the form and the
PDF extractor speak the same schema. Returns the same UploadResponse shape
as /upload (with chunks_indexed=0 -- there is no PDF to index), so the
frontend can reuse the eligibility + recommendation flow unchanged.
"""

import time
import uuid
from datetime import date

from fastapi import APIRouter, HTTPException

import store
import pdf_gen
import pdf_ingest
import rag_llamaindex
from settings import UPLOAD_DIR
from models import ApplicantProfile, UploadResponse

router = APIRouter(tags=["apply"])


@router.post("/apply", response_model=UploadResponse)
def apply(profile: ApplicantProfile) -> UploadResponse:
    """Save a hand-entered applicant profile and return its new id.

    Most validation lives on ApplicantProfile itself (counts and dollar
    amounts must be >= 0). Only cross-field / format checks live here.
    """
    t0 = time.perf_counter()
    if profile.desired_rent <= 0:
        raise HTTPException(400, "Please enter a desired rent above 0.")
    if profile.monthly_income < 0:
        raise HTTPException(400, "Monthly income cannot be negative.")
    if profile.credit_score is not None and not (
        300 <= profile.credit_score <= 850
    ):
        raise HTTPException(400, "Credit score must be between 300 and 850.")
    if profile.desired_move_in:
        try:
            date.fromisoformat(profile.desired_move_in)
        except ValueError:
            raise HTTPException(
                400, "Move-in date must look like 2026-08-01 (year-month-day)."
            )

    applicant_id = uuid.uuid4().hex[:12]

    # Generate a real application PDF from the form and run it through the
    # same index pipeline as uploads, so form applicants have a viewable PDF
    # and a working RAG chat — full parity with the upload/sample flow.
    UPLOAD_DIR.mkdir(exist_ok=True)
    pdf_path = UPLOAD_DIR / f"{applicant_id}.pdf"
    chunks = 0
    try:
        pdf_gen.generate_application_pdf(profile, str(pdf_path))
        text = pdf_ingest.extract_text(str(pdf_path))
        if text:
            chunks = rag_llamaindex.ingest(applicant_id, text)
    except Exception as exc:  # noqa: BLE001 - PDF is best-effort; never block apply
        print(f"Apply PDF generation/index failed: {type(exc).__name__}: {exc}")

    store.save_applicant(applicant_id, profile, chunks)
    store.log_event(
        endpoint="apply",
        applicant_id=applicant_id,
        latency_ms=(time.perf_counter() - t0) * 1000,
        source="form",
    )
    return UploadResponse(
        applicant_id=applicant_id,
        profile=profile,
        chunks_indexed=chunks,
        has_pdf=pdf_path.exists(),
    )
