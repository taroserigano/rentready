"""Turn an uploaded PDF into plain text.

Primary parser is PyMuPDF (fast, good with most PDFs). If it ever fails we
fall back to `unstructured`, which is better at messy forms and tables.
"""


def extract_text(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    try:
        import fitz  # PyMuPDF

        with fitz.open(pdf_path) as doc:
            return "\n".join(page.get_text() for page in doc).strip()
    except Exception as exc:  # noqa: BLE001
        print(f"PyMuPDF failed ({type(exc).__name__}); trying unstructured.")
        from unstructured.partition.auto import partition

        elements = partition(filename=pdf_path)
        return "\n".join(str(el) for el in elements).strip()
