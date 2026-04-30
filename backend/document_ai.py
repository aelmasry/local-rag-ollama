from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from backend.tools import get_logger

logger = get_logger("backend.document_ai")

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
DOCUMENT_SUFFIXES = {".pdf", ".txt", ".json"}
ALLOWED_SUFFIXES = IMAGE_SUFFIXES | DOCUMENT_SUFFIXES


def detect_file_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in DOCUMENT_SUFFIXES:
        return "document"
    return "unsupported"


def detect_document_type(text: str) -> str:
    normalized = (text or "").lower()
    id_keywords = (
        "passport",
        "national id",
        "identity",
        "id no",
        "id number",
        "date of birth",
        "expiry",
        "expires",
    )
    invoice_keywords = (
        "invoice",
        "invoice no",
        "invoice number",
        "bill to",
        "total amount",
        "subtotal",
        "tax",
        "vendor",
    )
    if any(k in normalized for k in id_keywords):
        return "id_card"
    if any(k in normalized for k in invoice_keywords):
        return "invoice"
    return "general_document"


def extract_structured_data(llm: Any, text: str, doc_type: str) -> dict[str, str] | None:
    if doc_type not in {"id_card", "invoice"}:
        return None

    if doc_type == "id_card":
        schema = {"name": "", "id_number": "", "expiry": ""}
    else:
        schema = {"total": "", "date": "", "vendor": ""}

    prompt = (
        "Extract structured data from the provided document text.\n"
        "Return STRICT JSON only. No markdown, no explanation.\n"
        f"Required keys: {list(schema.keys())}\n"
        'If missing, return empty string "". \n\n'
        f"Document text:\n{text[:6000]}"
    )
    response = llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        structured = {k: str(parsed.get(k, "") or "") for k in schema}
        return structured
    except json.JSONDecodeError:
        logger.warning("structured_extraction_parse_failed doc_type=%s raw=%s", doc_type, cleaned[:500])
        return schema


@dataclass
class ProcessedDocument:
    source_file: str
    file_type: str
    doc_type: str
    structured_data: dict[str, str] | None
    text_preview: str
    retrieval_ready: bool


def summarize_docs_for_classification(docs: list[Document]) -> str:
    text = "\n".join(doc.page_content for doc in docs)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:8000]
