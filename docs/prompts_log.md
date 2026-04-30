# Prompts Log

## Document Metadata

- Author: Ali Salem
- Last Updated: 2026-04-30

## Prompts Used in Project

This file documents prompt templates currently implemented in the codebase and operational prompt guidance used around the project.

---

## 1. System Prompts

### 1.1 Structured JSON Answer Prompt

**Location:** `backend/document_ai.py` -> `build_structured_answer_prompt`

**Purpose:** Answer user queries strictly from cached structured JSON fields (ID/invoice flows), without inventing data.

```text
You are answering strictly from structured JSON data.
Use ONLY the values in the JSON object.
If requested value is empty or missing, reply exactly: NOT_FOUND.
Do not add assumptions.
```

**When used:** `/ask` route when retrieved source maps to a structured document (`id` or `invoice`) and cache is present.

---

## 2. RAG Prompts

### 2.1 CV QA Prompt

**Location:** `backend/document_ai.py` -> `build_cv_qa_prompt`

**Purpose:** Answer CV questions from retrieved context with strict anti-hallucination behavior.

```text
You are a CV question-answering assistant.
Answer only from the context.
If the answer is not present, reply exactly: NOT_FOUND.
Do not hallucinate or infer beyond explicit evidence.
```

**When used:** `/ask` for documents classified as `cv` (RAG mode).

### 2.2 General QA Prompt

**Location:** `backend/document_ai.py` -> `build_general_qa_prompt`

**Purpose:** Generic QA over retrieved chunks for non-structured, non-CV documents.

```text
You are a document QA assistant.
Use only the provided context and do not hallucinate.
If the answer cannot be found, reply exactly: NOT_FOUND.
```

**When used:** `/ask` for `general` documents (RAG mode).

### 2.3 Retrieval Query Rewrite Rule (Prompt-like Retrieval Hint)

**Location:** `backend/rag.py` -> `rewrite_query_for_retrieval`

**Behavior:** Name-like queries are rewritten to improve CV retrieval focus:

```text
What is the full name of the person in this CV?
```

**When used:** Before retrieval for vague name questions (`name`, `my name`, `full name`, `who am i`).

---

## 3. OCR / Extraction Prompts

### 3.1 Structured Extraction Prompt (ID / Invoice)

**Location:** `backend/document_ai.py` -> `extract_structured_data`

**Purpose:** Convert extracted document text into strict JSON with fixed schema keys.

Core instructions:

```text
Extract structured data from the provided document text.
Return STRICT JSON only. No markdown, no explanation.
Required keys: [...]
If missing, return empty string "".
```

**Schemas used:**

- **ID:** `name`, `id_number`, `expiry`
- **Invoice:** `total`, `date`, `vendor`

**When used:** During `/upload` for documents classified as `id` or `invoice` (run once, then cached).

### 3.2 OCR Non-LLM Text Processing Rules

**Location:** `backend/ocr.py` -> `clean_ocr_text`

This is not an LLM prompt, but a deterministic extraction-cleaning stage:

- remove empty lines
- normalize whitespace
- drop junk single-character punctuation lines
- collapse repeated blank lines

---

## 4. Cursor Prompts

The repository does not contain a dedicated in-repo prompt library for Cursor (for setup/RAG/OCR/Docker) as source files.

Operational guidance currently appears in:

- `README.md` (setup commands, runtime flow, troubleshooting)
- chat/session-level instructions (outside repository files)

If needed, create a future prompt catalog under `docs/` (for example `docs/cursor_workflows.md`) to formalize reusable Cursor guidance for:

- environment setup
- OCR troubleshooting
- RAG debugging
- Docker orchestration
- validation workflows

---

## 5. Notes

- Prompt routing is dynamic and type-aware via document classification (`id`, `invoice`, `cv`, `general`).
- Prompt usage is logged (prompt name and output preview) in API flow.
- Structured extraction is executed once on upload for structured types and reused from cache.
- Anti-hallucination behavior is enforced by explicit `NOT_FOUND` fallback rules in QA and structured-answer prompts.
- Future enhancement: add versioned prompt templates and evaluation metrics per prompt variant.
