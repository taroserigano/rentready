"""PDF RAG with LlamaIndex + ChromaDB.

Responsibilities:
  - index an application's text into a shared Chroma collection
  - answer questions about a specific application (RAG, filtered by id)
  - extract a structured ApplicantProfile from the text

When no Anthropic key is set, query() returns the retrieved context
(retrieve-only) and extract_profile() uses a regex heuristic, so the whole
flow still works offline.
"""

import json
import re
from functools import lru_cache

from settings import CHROMA_DIR, settings, get_embeddings
from llm import get_llamaindex_llm
from models import ApplicantProfile

COLLECTION = "applications"


@lru_cache(maxsize=1)
def _index():
    import chromadb
    from llama_index.core import StorageContext, VectorStoreIndex, Settings
    from llama_index.vector_stores.chroma import ChromaVectorStore

    Settings.embed_model = get_embeddings()
    Settings.llm = get_llamaindex_llm()  # may be None -> retrieve-only

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )


def ingest(applicant_id: str, text: str) -> int:
    """Chunk the text, tag each chunk with the applicant id, store it."""
    from llama_index.core import Document
    from llama_index.core.node_parser import SentenceSplitter

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
    doc = Document(text=text, metadata={"applicant_id": applicant_id})
    nodes = splitter.get_nodes_from_documents([doc])
    _index().insert_nodes(nodes)
    return len(nodes)


def _filters(applicant_id: str):
    from llama_index.core.vector_stores import (
        FilterOperator,
        MetadataFilter,
        MetadataFilters,
    )

    return MetadataFilters(
        filters=[
            MetadataFilter(
                key="applicant_id",
                value=applicant_id,
                operator=FilterOperator.EQ,
            )
        ]
    )


def retrieve_contexts(applicant_id: str, question: str) -> list:
    """Return the FULL retrieved chunk texts for one applicant.

    Used by RAGAS, which needs the complete context (not the truncated UI
    snippets) to verify the answer's faithfulness.
    """
    retriever = _index().as_retriever(
        similarity_top_k=settings.retriever_k, filters=_filters(applicant_id)
    )
    return [n.node.get_content() for n in retriever.retrieve(question)]


def query(applicant_id: str, question: str) -> dict:
    """Answer a question about one applicant's documents."""
    k = settings.retriever_k
    index = _index()

    if get_llamaindex_llm() is not None:
        engine = index.as_query_engine(
            similarity_top_k=k, filters=_filters(applicant_id)
        )
        resp = engine.query(question)
        sources = [
            n.node.get_content()[:160].replace("\n", " ") + "…"
            for n in resp.source_nodes
        ]
        return {"answer": str(resp), "source": "anthropic", "sources": sources}

    # No LLM: retrieve-only mock answer.
    retriever = index.as_retriever(
        similarity_top_k=k, filters=_filters(applicant_id)
    )
    nodes = retriever.retrieve(question)
    context = "\n\n".join(n.node.get_content() for n in nodes)
    sources = [
        n.node.get_content()[:160].replace("\n", " ") + "…" for n in nodes
    ]
    answer = (
        "[mock answer] No ANTHROPIC_API_KEY set. Top retrieved context for "
        f"'{question}':\n\n{context[:500]}…"
    )
    return {"answer": answer, "source": "mock", "sources": sources}


_EXTRACTION_PROMPT = """You are extracting structured data from a rental \
application. Return ONLY valid JSON (no prose) with these keys:
name (string), monthly_income (number), desired_rent (number),
credit_score (integer or null), employment_status (string),
has_pets (boolean), preferred_area (string),
wanted_amenities (array of strings), bedrooms_wanted (integer or null),
bathrooms_wanted (number or null),
bath_type_wanted (one of "full", "shower_only", "any"),
min_square_feet (integer or null),
needs_balcony (boolean), needs_parking (boolean),
needs_in_unit_laundry (boolean), furnished_wanted (boolean),
lease_term_wanted (integer months or null).
If the text mentions them, also include: employer (string),
job_title (string), monthly_debt_payments (number),
evictions_count (integer), bankruptcies_count (integer),
household_size (integer), pet_count (integer),
guarantor_available (boolean), desired_move_in (ISO date string).
Omit any of these optional keys when the text does not mention them.

Application text:
---
{text}
---
JSON:"""


def extract_profile(text: str) -> ApplicantProfile:
    """Pull a structured profile out of the application text."""
    llm = get_llamaindex_llm()
    if llm is not None:
        try:
            raw = llm.complete(_EXTRACTION_PROMPT.format(text=text[:6000]))
            data = _first_json(str(raw))
            if data:
                return ApplicantProfile(**data)
        except Exception as exc:  # noqa: BLE001
            print(f"LLM extraction failed ({type(exc).__name__}); heuristic.")
    return _heuristic_profile(text)


def _first_json(s: str) -> dict:
    match = re.search(r"\{.*\}", s, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _heuristic_profile(text: str) -> ApplicantProfile:
    """Regex-based extraction so things work without an LLM."""

    def num(pattern: str):
        m = re.search(pattern, text, re.IGNORECASE)
        if not m:
            return None
        return float(m.group(1).replace(",", ""))

    def has(pattern: str) -> bool:
        return bool(re.search(pattern, text, re.IGNORECASE))

    def pref(keyword: str) -> bool:
        """A preference is True unless explicitly labelled 'No'.

        Handles "Balcony: No" / "Parking: Yes - ..." style fields without
        false-positiving on the label word itself.
        """
        if re.search(rf"(?:{keyword})\s*:?\s*no\b", text, re.IGNORECASE):
            return False
        return has(keyword)

    # Pets needs special handling because "Pets: No" contains the word "pet".
    pets_m = re.search(r"pets?\s*:?\s*(yes|no)", text, re.IGNORECASE)
    if pets_m:
        has_pets = pets_m.group(1).lower() == "yes"
    else:
        has_pets = has(r"\b(dog|cat)\b")

    name_m = re.search(r"name[:\s]+([A-Za-z .'-]{2,40})", text, re.IGNORECASE)
    area_m = re.search(
        r"preferred (?:area|location)[:\s]+([A-Za-z .'-]{2,40})",
        text,
        re.IGNORECASE,
    )
    income = num(r"(?:monthly income|income)[:\s$]*([\d,]+)") or 0.0
    rent = num(r"(?:desired rent|rent)[:\s$]*([\d,]+)") or 0.0
    credit = num(r"credit score[:\s]*([\d]{3})")
    beds = num(r"bedrooms?[:\s]*([\d])")
    baths = num(r"bathrooms?[:\s]*([\d.]+)")
    sqft = num(r"(?:square ?feet|sq ?ft|minimum size)[:\s]*([\d,]+)")
    lease = num(r"lease(?:\s*term)?[:\s]*([\d]+)")

    bath_type = "any"
    if has(r"full bath"):
        bath_type = "full"
    elif has(r"shower[- ]only"):
        bath_type = "shower_only"

    return ApplicantProfile(
        name=name_m.group(1).strip() if name_m else "Unknown",
        monthly_income=income,
        desired_rent=rent,
        credit_score=int(credit) if credit else None,
        employment_status="employed" if has(r"employ") else "unknown",
        bedrooms_wanted=int(beds) if beds else None,
        bathrooms_wanted=baths,
        bath_type_wanted=bath_type,
        min_square_feet=int(sqft) if sqft else None,
        has_pets=has_pets,
        needs_balcony=pref(r"balcony"),
        needs_parking=pref(r"parking|garage"),
        needs_in_unit_laundry=pref(r"in[- ]unit laundry|washer"),
        furnished_wanted=pref(r"furnished"),
        preferred_area=area_m.group(1).strip() if area_m else "",
        wanted_amenities=_find_amenities(text),
        lease_term_wanted=int(lease) if lease else None,
    )


# Community amenities (parking/balcony/laundry are structured fields now).
_KNOWN_AMENITIES = [
    "Gym",
    "Pool",
    "Pet Park",
    "Rooftop Deck",
    "Bike Storage",
    "Concierge",
    "Playground",
]


def _find_amenities(text: str) -> list:
    found = []
    low = text.lower()
    for amenity in _KNOWN_AMENITIES:
        if amenity.lower() in low:
            found.append(amenity)
    return found
