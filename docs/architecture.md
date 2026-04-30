# Document AI System Architecture

## Document Metadata

- Author: Ali Salem
- Last Updated: 2026-04-30

## 1. Project Overview

This project is a local-first Document AI system that supports:

- Retrieval-Augmented Generation (RAG) for text-heavy documents.
- OCR for image uploads.
- Structured extraction for semi-structured documents (ID and invoice).
- Interactive usage through a FastAPI backend and Streamlit UI.

Core capabilities:

- Upload and process `pdf`, `txt`, `json`, `png`, `jpg`, `jpeg`.
- Extract text (native loaders or OCR), chunk, embed, and index into FAISS.
- Detect document type (`id`, `invoice`, `cv`, `general`) and route query behavior.
- Cache structured JSON results per uploaded file for reuse in later queries.

---

## 2. System Architecture

High-level components:

- **UI (`ui/app.py`)**
  - Streamlit front-end for health checks, file upload, and question answering.
- **API (`backend/main.py`)**
  - FastAPI service exposing `/health`, `/upload`, `/ask`.
  - Orchestrates ingestion, detection, extraction, and response strategy.
- **RAG Engine (`backend/rag.py`)**
  - LangChain loaders, chunking, embeddings, retrieval, and answer generation helpers.
- **OCR Module (`backend/ocr.py`)**
  - Local OCR using `pytesseract` + `Pillow`.
- **Document Intelligence Layer (`backend/document_ai.py`)**
  - File/document type detection, prompt builders, structured extraction logic.
- **Vector Database (`vectorstore/faiss_index`)**
  - Persisted FAISS index for semantic retrieval.
- **Local LLM Runtime (Ollama)**
  - Generation model: `llama3`.
  - Embedding model: `nomic-embed-text`.

---

## 3. Workflow

### A) Text Document Flow (PDF/TXT/JSON)

1. User uploads file through Streamlit (`/upload`).
2. API validates and stores file under `data/uploads`.
3. `RagService` loads text:
   - PDF via `PyPDFLoader`
   - TXT via `TextLoader`
   - JSON via custom flattening (`json_file_to_documents`)
4. Document text is chunked (`RecursiveCharacterTextSplitter`).
5. Chunks are embedded with `OllamaEmbeddings(nomic-embed-text)`.
6. Chunks are indexed into FAISS (append to existing index if present).
7. On query (`/ask`), retriever returns top-k chunks.
8. System chooses prompt strategy (CV/general) and generates answer.

### B) Image Flow

1. User uploads image (`png/jpg/jpeg`).
2. API stores file and ingestion calls OCR path in `RagService`.
3. `ocr.extract_text_from_image()` runs Tesseract.
4. OCR text is cleaned (`clean_ocr_text`) and converted to `Document`.
5. Text is chunked, embedded, and indexed in FAISS.
6. API classifies document type from extracted text.
7. If type is `id`/`invoice`, structured extraction is executed and cached.
8. On query, system answers from cached JSON (structured mode) or RAG.

### C) Structured Documents (ID / Invoice)

1. Upload (image or text-like source).
2. Extract text (OCR or loader path).
3. Detect type using keyword-based classification.
4. Run one-time structured extraction prompt:
   - ID schema: `name`, `id_number`, `expiry`
   - Invoice schema: `total`, `date`, `vendor`
5. Parse and store strict JSON in cache keyed by source file.
6. Future queries referencing that file use cached JSON prompt-first answering.

---

## 4. Folder Structure

### `/backend`

- `main.py`: FastAPI app, endpoints, pipeline orchestration, in-memory caches.
- `rag.py`: document loading, chunking, embedding, FAISS indexing, retrieval utilities.
- `ocr.py`: OCR extraction and cleaning.
- `document_ai.py`: classification, structured extraction, prompt builders.
- `tools.py`: logging setup, retry helper, JSON flattening utility.
- `config.py`: paths, model names, chunking/retrieval configuration.
- `validation_tests.py`: script for scenario-based endpoint validation.
- `__init__.py`: package marker.

### `/ui`

- `app.py`: Streamlit interface for health check, upload, and ask flow.

### `/data`

- `uploads/`: persisted uploaded files used by ingestion pipeline.
- other data artifacts (for example sample JSON).

### `/vectorstore`

- `faiss_index/`: persisted FAISS index files (`index.faiss`, metadata store files).

---

## 5. Component Breakdown

### `backend/main.py` (API Layer)

- Defines FastAPI app and CORS.
- Implements file suffix detection and upload storage safeguards.
- Coordinates ingestion via `RagService.ingest_files`.
- Performs document type detection and structured extraction on upload.
- Maintains in-memory caches:
  - `structured_cache_by_file`
  - `doc_type_by_file`
  - `processed_documents`
- Routes `/ask` to:
  - structured JSON answering for ID/invoice when cache exists
  - RAG QA for CV/general documents

### `backend/rag.py` (Retrieval Logic)

- Initializes embeddings + LLM + splitter.
- Loads documents by suffix (PDF/TXT/JSON/Image OCR).
- Adds high-priority header chunk per document.
- Creates or appends to FAISS index.
- Retrieves top-k relevant chunks and normalizes text.
- Exposes reusable methods for retrieval and answer generation.

### `backend/ocr.py` (Image Processing)

- Checks Tesseract availability.
- Runs OCR from image bytes.
- Cleans noisy OCR output (whitespace normalization, junk-line removal).
- Logs raw and cleaned OCR previews for debugging.

### `backend/tools.py`

- Central logger factory (`get_logger`).
- Retry decorator for transient errors.
- JSON flattening and conversion to LangChain `Document` objects.

### `backend/config.py`

- Central config for:
  - data/vector paths
  - Ollama URL
  - LLM and embedding model names
  - chunk size/overlap and retriever k
  - document header chunk length

---

## 6. Class Diagram (Text-based)

```text
User
  -> Streamlit UI (ui/app.py)
    -> FastAPI API (backend/main.py)
      -> RagService (backend/rag.py)
        -> Loaders / OCR module (backend/ocr.py)
        -> Text Splitter
        -> Ollama Embeddings (nomic-embed-text)
        -> FAISS Vector Store
        -> Retriever (k=5)
      -> Ollama Chat LLM (llama3)
      -> Document AI Layer (backend/document_ai.py)
         -> Type Detection
         -> Structured Extraction + Cache
```

Simplified path:

```text
User -> UI -> API -> (DocumentAI + RAG) -> FAISS / Ollama -> Answer
```

---

## 7. Data Flow Diagram (Text)

```text
Upload Request
  -> File normalization + type detection
  -> Saved to data/uploads
  -> Text extraction (loader or OCR)
  -> Classification (id/invoice/cv/general)
  -> (optional) Structured extraction -> JSON cache
  -> Chunking + embedding + FAISS index update

Ask Request
  -> Retrieve relevant chunks from FAISS
  -> Identify matched source file/type
  -> if id/invoice + cached JSON: answer from structured JSON prompt
  -> else: answer from CV/general RAG prompt with retrieved context
  -> Return answer + retrieved chunks (+ structured data when relevant)
```

---

## 8. Decision Logic

Decision points implemented by current system:

1. **OCR vs non-OCR**
   - File suffix in image set (`png/jpg/jpeg`) -> OCR route.
   - Otherwise text loader route.

2. **Structured extraction vs RAG QA**
   - After text extraction, classify document type.
   - If `id` or `invoice`, run one-time structured extraction and cache result.
   - On ask:
     - if retrieved source maps to `id`/`invoice` and cache exists -> structured answer mode
     - else -> RAG mode (CV prompt or general prompt).

---

## 9. Technologies Used

- **Ollama** (local model serving)
- **LangChain** (loaders, retriever/vector workflows)
- **FastAPI** (backend API)
- **FAISS** (local vector database)
- **Streamlit** (web UI)
- **pytesseract** + **Pillow** (OCR)
- **PyPDF** via `PyPDFLoader` (PDF parsing)

---

## 10. Known Issues & Limitations

- OCR quality depends heavily on image quality and layout.
- Scanned PDFs are not explicitly routed through a dedicated OCR-for-PDF pipeline.
- Retriever can miss relevant context for complex tables/layout-heavy docs.
- Document type detection is keyword-based; may misclassify edge cases.
- Structured cache is in-memory (not persisted across backend restarts).
- FAISS loading currently uses dangerous deserialization flag (security trade-off).
- CORS is permissive for local/dev usage.

---

## 11. Future Improvements

- Replace keyword classification with model-based document classifier.
- Add hybrid retrieval (BM25 + vector) and reranking.
- Add persistent metadata/cache store (SQLite/Postgres/Redis).
- Add scanned-PDF OCR path with page-level image extraction.
- Introduce structured extraction validation with confidence scoring.
- Add evaluation harness (retrieval quality, answer faithfulness, extraction accuracy).
- Add regression test suite and dataset-driven benchmark workflow.
