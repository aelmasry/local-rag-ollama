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
DOC_TYPE_ID = "id"
DOC_TYPE_INVOICE = "invoice"
DOC_TYPE_CV = "cv"
DOC_TYPE_GENERAL = "general"


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
    cv_keywords = (
        "curriculum vitae",
        "resume",
        "experience",
        "education",
        "skills",
        "work history",
        "linkedin",
    )
    if any(k in normalized for k in id_keywords):
        return DOC_TYPE_ID
    if any(k in normalized for k in invoice_keywords):
        return DOC_TYPE_INVOICE
    if any(k in normalized for k in cv_keywords):
        return DOC_TYPE_CV
    return DOC_TYPE_GENERAL


def get_prompt_name(doc_type: str, mode: str) -> str:
    return f"{doc_type}_{mode}"


def build_cv_qa_prompt(question: str, context: str) -> str:
    return (
        "You are a CV question-answering assistant.\n"
        "Answer only from the context.\n"
        "If the answer is not present, reply exactly: NOT_FOUND.\n"
        "Do not hallucinate or infer beyond explicit evidence.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )


def build_general_qa_prompt(question: str, context: str) -> str:
    return (
        "You are a document QA assistant.\n"
        "Use only the provided context and do not hallucinate.\n"
        "If the answer cannot be found, reply exactly: NOT_FOUND.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )


def extract_structured_data(llm: Any, text: str, doc_type: str) -> dict[str, str] | None:
    if doc_type not in {DOC_TYPE_ID, DOC_TYPE_INVOICE}:
        return None

    if doc_type == DOC_TYPE_ID:
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
        logger.info("structured_extraction_output doc_type=%s output=%s", doc_type, structured)
        return structured
    except json.JSONDecodeError:
        logger.warning("structured_extraction_parse_failed doc_type=%s raw=%s", doc_type, cleaned[:500])
        return schema


def build_structured_answer_prompt(question: str, doc_type: str, structured_data: dict[str, str]) -> str:
    return (
        "You are answering strictly from structured JSON data.\n"
        "Use ONLY the values in the JSON object.\n"
        "If requested value is empty or missing, reply exactly: NOT_FOUND.\n"
        "Do not add assumptions.\n\n"
        f"Document type: {doc_type}\n"
        f"Question: {question}\n"
        f"JSON: {json.dumps(structured_data, ensure_ascii=True)}\n\n"
        "Answer:"
    )


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
