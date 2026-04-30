from pathlib import Path
import re

import ollama
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import EMBED_MODEL, LLM_MODEL, OLLAMA_BASE_URL, UPLOAD_DIR
from backend.document_ai import (
    ALLOWED_SUFFIXES,
    DOC_TYPE_CV,
    DOC_TYPE_GENERAL,
    DOC_TYPE_ID,
    DOC_TYPE_INVOICE,
    ProcessedDocument,
    build_cv_qa_prompt,
    build_general_qa_prompt,
    build_structured_answer_prompt,
    detect_document_type,
    detect_file_type,
    extract_structured_data,
    get_prompt_name,
    summarize_docs_for_classification,
)
from backend.ocr import tesseract_available
from backend.rag import RagService
from backend.tools import get_logger

logger = get_logger("backend.api")
app = FastAPI(title="Local RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_service = RagService()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
processed_documents: list[ProcessedDocument] = []
structured_cache_by_file: dict[str, dict[str, str]] = {}
doc_type_by_file: dict[str, str] = {}


def _normalize_filename(name: str) -> str:
    """ASCII-safe name for Path; map common Unicode full-stop lookalikes to '.'."""
    n = (name or "").strip()
    for ch in ("\uff0e", "\u3002", "\ufe52", "\uff61"):
        n = n.replace(ch, ".")
    return n


def _detect_suffix(filename: str, content: bytes, content_type: str | None) -> str:
    """Extension from filename, then Content-Type, then file magic (handles odd Unicode in names)."""
    suffix = Path(_normalize_filename(filename)).suffix.lower()
    if suffix == ".jfif":
        suffix = ".jpeg"
    if suffix in ALLOWED_SUFFIXES:
        return suffix

    ct = (content_type or "").split(";")[0].strip().lower()
    if ct == "image/png":
        return ".png"
    if ct in {"image/jpeg", "image/jpg"}:
        return ".jpeg"

    if len(content) >= 8 and content.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if len(content) >= 3 and content.startswith(b"\xff\xd8\xff"):
        return ".jpeg"
    if content.startswith(b"%PDF"):
        return ".pdf"

    return ""


class AskRequest(BaseModel):
    question: str


def _unique_path(stem: str, suffix: str) -> Path:
    candidate = UPLOAD_DIR / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate
    index = 1
    while True:
        candidate = UPLOAD_DIR / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


@app.get("/health")
def health() -> dict:
    try:
        models = ollama.Client(host=OLLAMA_BASE_URL).list().models
        model_names = [m.model for m in models]
    except Exception as exc:  # noqa: BLE001
        logger.exception("health_check_failed error=%s", exc)
        raise HTTPException(status_code=503, detail=f"Ollama unavailable: {exc}") from exc

    return {
        "status": "ok",
        "ollama_url": OLLAMA_BASE_URL,
        "llm_model": LLM_MODEL,
        "embed_model": EMBED_MODEL,
        "models_available": model_names,
        "ocr_tesseract": "ok" if tesseract_available() else "unavailable",
    }


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    saved_paths: list[Path] = []
    try:
        for uploaded in files:
            content = await uploaded.read()
            raw_name = uploaded.filename or "uploaded_file"
            suffix = _detect_suffix(raw_name, content, uploaded.content_type)
            if suffix not in ALLOWED_SUFFIXES:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unsupported file type for {raw_name!r} "
                        f"(declared suffix={Path(_normalize_filename(raw_name)).suffix!r}, "
                        f"content_type={uploaded.content_type!r}). "
                        "Allowed: pdf, txt, json, png, jpg, jpeg."
                    ),
                )

            safe_name = Path(_normalize_filename(raw_name)).name
            stem = Path(safe_name).stem or "uploaded_file"
            save_path = _unique_path(stem, suffix)
            save_path.write_bytes(content)
            saved_paths.append(save_path)
            logger.info(
                "file_uploaded name=%s bytes=%d file_type=%s",
                save_path.name,
                len(content),
                detect_file_type(save_path),
            )

        ingest_stats = rag_service.ingest_files(saved_paths)
        for path in saved_paths:
            docs = rag_service.load_documents_for_file(path)
            classification_text = summarize_docs_for_classification(docs)
            doc_type = detect_document_type(classification_text)
            structured_data = extract_structured_data(rag_service.llm, classification_text, doc_type)
            doc_type_by_file[path.name] = doc_type
            if structured_data is not None:
                structured_cache_by_file[path.name] = structured_data
            prompt_used = get_prompt_name(
                doc_type,
                "structured_extraction" if doc_type in {DOC_TYPE_ID, DOC_TYPE_INVOICE} else "rag_qa",
            )
            logger.info(
                "document_type_detected file=%s file_type=%s doc_type=%s prompt_used=%s ocr_preview=%s output=%s",
                path.name,
                detect_file_type(path),
                doc_type,
                prompt_used,
                re.sub(r"\s+", " ", classification_text[:250]),
                structured_data,
            )
            processed_documents.append(
                ProcessedDocument(
                    source_file=path.name,
                    file_type=detect_file_type(path),
                    doc_type=doc_type,
                    structured_data=structured_data,
                    text_preview=classification_text[:500],
                    retrieval_ready=True,
                )
            )
        return {
            "message": "Files uploaded and indexed successfully.",
            "files": [p.name for p in saved_paths],
            "processed_documents": [
                {
                    "source_file": item.source_file,
                    "file_type": item.file_type,
                    "doc_type": item.doc_type,
                    "structured_data": item.structured_data,
                }
                for item in processed_documents[-len(saved_paths) :]
            ],
            **ingest_stats,
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("upload_failed error=%s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask")
def ask(payload: AskRequest) -> dict:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")
    try:
        retrieved = rag_service.retrieve(question)
        retrieved_files = {
            (d.metadata.get("source_file") or Path(str(d.metadata.get("source", ""))).name)
            for d in retrieved["docs"]
        }
        matched_structured = next(
            (
                d
                for d in reversed(processed_documents)
                if d.source_file in retrieved_files
                and d.doc_type in {DOC_TYPE_ID, DOC_TYPE_INVOICE}
                and d.structured_data
            ),
            None,
        )
        # Prefer cached structured extraction for known document forms.
        if matched_structured:
            cached_structured = structured_cache_by_file.get(matched_structured.source_file)
        else:
            cached_structured = None
        if matched_structured and cached_structured:
            prompt_name = get_prompt_name(matched_structured.doc_type, "qa_from_cached_json")
            prompt = build_structured_answer_prompt(
                question=question,
                doc_type=matched_structured.doc_type,
                structured_data=cached_structured,
            )
            logger.info("ask_prompt_used prompt=%s file=%s", prompt_name, matched_structured.source_file)
            answer = rag_service.llm.invoke(prompt)
            answer_text = answer.content if hasattr(answer, "content") else str(answer)
            logger.info("ask_output mode=structured_plus_rag output=%s", answer_text[:500])
            return {
                "mode": "structured_plus_rag",
                "answer": answer_text,
                "structured_data": cached_structured,
                "document_type": matched_structured.doc_type,
                "retrieved_chunks": [
                    {"metadata": d.metadata, "content": d.page_content[:1200]}
                    for d in retrieved["docs"]
                ],
                "k": retrieved["k"],
            }
        context = "\n\n---\n\n".join(d.page_content for d in retrieved["docs"])
        retrieved_type = next(
            (
                doc_type_by_file.get(
                    d.metadata.get("source_file") or Path(str(d.metadata.get("source", ""))).name,
                    DOC_TYPE_GENERAL,
                )
                for d in retrieved["docs"]
            ),
            DOC_TYPE_GENERAL,
        )
        if retrieved_type == DOC_TYPE_CV:
            prompt_name = get_prompt_name(DOC_TYPE_CV, "rag_qa")
            prompt = build_cv_qa_prompt(question=retrieved["original_question"], context=context)
        else:
            prompt_name = get_prompt_name(DOC_TYPE_GENERAL, "rag_qa")
            prompt = build_general_qa_prompt(question=retrieved["original_question"], context=context)
        logger.info("ask_prompt_used prompt=%s", prompt_name)
        answer = rag_service.llm.invoke(prompt)
        answer_text = answer.content if hasattr(answer, "content") else str(answer)
        logger.info("ask_output mode=rag output=%s", answer_text[:500])
        return {
            "mode": "rag",
            "answer": answer_text,
            "original_question": retrieved["original_question"],
            "retrieval_query": retrieved["retrieval_query"],
            "query_rewritten": retrieved["query_rewritten"],
            "retrieved_chunks": [
                {"metadata": d.metadata, "content": d.page_content[:1200]}
                for d in retrieved["docs"]
            ],
            "structured_data": None,
            "document_type": retrieved_type,
            "k": retrieved["k"],
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("ask_failed error=%s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
