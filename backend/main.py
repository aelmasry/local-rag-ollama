from pathlib import Path

import ollama
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import EMBED_MODEL, LLM_MODEL, OLLAMA_BASE_URL, UPLOAD_DIR
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

ALLOWED_SUFFIXES = frozenset({".pdf", ".txt", ".json", ".png", ".jpg", ".jpeg"})


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
            save_path = UPLOAD_DIR / f"{stem}{suffix}"
            save_path.write_bytes(content)
            saved_paths.append(save_path)
            logger.info("file_uploaded name=%s bytes=%d", save_path.name, len(content))

        ingest_stats = rag_service.ingest_files(saved_paths)
        return {
            "message": "Files uploaded and indexed successfully.",
            "files": [p.name for p in saved_paths],
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
        return rag_service.ask(question)
    except Exception as exc:  # noqa: BLE001
        logger.exception("ask_failed error=%s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
