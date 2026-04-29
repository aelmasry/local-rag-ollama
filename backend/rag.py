from pathlib import Path
import re

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document

from backend.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENT_HEADER_CHAR_LIMIT,
    EMBED_MODEL,
    LLM_MODEL,
    OLLAMA_BASE_URL,
    RETRIEVER_K,
    VECTORSTORE_DIR,
)
from backend import ocr as ocr_module
from backend.tools import get_logger, json_file_to_documents, retry

logger = get_logger("backend.rag")

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
CV_NAME_REWRITE = "What is the full name of the person in this CV?"


def rewrite_query_for_retrieval(question: str) -> tuple[str, bool]:
    """Rewrite vague queries so retrieval matches CV-style content."""
    q_strip = question.strip()
    q_lower = q_strip.lower()
    if (
        re.search(r"\bname\b", q_lower)
        or "my name" in q_lower
        or "full name" in q_lower
        or "who am i" in q_lower
    ):
        return CV_NAME_REWRITE, True
    return q_strip, False


def _log_retrieved_chunks(docs: list, label: str = "retrieval") -> None:
    for i, doc in enumerate(docs, start=1):
        preview = doc.page_content[:400].replace("\n", " ")
        logger.info(
            "[%s] chunk=%d metadata=%s preview=%s",
            label,
            i,
            doc.metadata,
            preview,
        )


class RagService:
    def __init__(self) -> None:
        self.embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
        self.llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

    def _load_documents(self, file_path: Path):
        suffix = file_path.suffix.lower()
        logger.info("loading_file path=%s type=%s", file_path, suffix)

        if suffix == ".pdf":
            return PyPDFLoader(str(file_path)).load()
        if suffix == ".txt":
            return TextLoader(str(file_path), encoding="utf-8").load()
        if suffix == ".json":
            return json_file_to_documents(file_path)
        if suffix in IMAGE_SUFFIXES:
            data = file_path.read_bytes()
            text = ocr_module.extract_text_from_image(data)
            return [
                Document(
                    page_content=text,
                    metadata={
                        "source": str(file_path),
                        "source_file": file_path.name,
                        "content_source": "ocr_image",
                    },
                )
            ]

        raise ValueError(f"Unsupported file type: {suffix}")

    @retry(attempts=3, delay_seconds=1.0)
    def ingest_files(self, file_paths: list[Path]) -> dict:
        all_docs: list[Document] = []
        for path in file_paths:
            file_docs = self._load_documents(path)
            if not file_docs:
                continue
            full_text = "\n".join(d.page_content for d in file_docs)
            normalized = re.sub(r"\s+", " ", full_text).strip()
            head = normalized[:DOCUMENT_HEADER_CHAR_LIMIT]
            if head:
                base_meta = dict(file_docs[0].metadata)
                base_meta["chunk_role"] = "document_header"
                base_meta["high_priority"] = True
                base_meta["source_file"] = path.name
                header_doc = Document(
                    page_content=(
                        "[DOCUMENT_HEADER — first "
                        f"{DOCUMENT_HEADER_CHAR_LIMIT} characters — high priority]\n"
                        f"{head}"
                    ),
                    metadata=base_meta,
                )
                all_docs.append(header_doc)
                logger.info(
                    "document_header_added file=%s chars=%d",
                    path.name,
                    len(head),
                )
            all_docs.extend(file_docs)

        if not all_docs:
            raise ValueError("No documents were loaded from uploaded files.")

        logger.info("chunking documents=%d", len(all_docs))
        chunks = self.splitter.split_documents(all_docs)
        logger.info("chunking_done chunks=%d", len(chunks))

        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("embedding model=%s", EMBED_MODEL)
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        vectorstore.save_local(str(VECTORSTORE_DIR))
        logger.info("vectorstore_saved path=%s", VECTORSTORE_DIR)
        return {"documents": len(all_docs), "chunks": len(chunks)}

    @retry(attempts=3, delay_seconds=1.0)
    def ask(self, question: str) -> dict:
        if not VECTORSTORE_DIR.exists():
            raise FileNotFoundError("Vectorstore not found. Upload and index files first.")

        original_question = question.strip()
        retrieval_query, rewritten = rewrite_query_for_retrieval(original_question)
        if rewritten:
            logger.info(
                "query_rewrite original=%r retrieval=%r",
                original_question,
                retrieval_query,
            )

        logger.info("retrieval question=%s", retrieval_query)
        db = FAISS.load_local(
            str(VECTORSTORE_DIR),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        retriever = db.as_retriever(search_kwargs={"k": RETRIEVER_K})
        docs = retriever.invoke(retrieval_query)

        # Normalize PDF-extracted text (often contains broken spaces/newlines).
        normalized_docs = []
        for doc in docs:
            cleaned = re.sub(r"\s+", " ", doc.page_content).strip()
            doc.page_content = cleaned
            normalized_docs.append(doc)
        docs = normalized_docs

        _log_retrieved_chunks(docs, label="debug")

        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        has_ocr = any(d.metadata.get("content_source") == "ocr_image" for d in docs)
        id_like = has_ocr or bool(
            re.search(
                r"(?i)(passport|national\s*id|identity|id\s*no|document\s*no|"
                r"expir|expires|date\s*of\s*birth|\bdoi\b)",
                context,
            )
        )

        id_instructions = ""
        if id_like:
            id_instructions = (
                "\nThe context may be an ID card, passport, or similar document. "
                "If so, end your answer with a short structured block (only fields "
                "clearly present in the text; write \"not shown\" if absent):\n"
                "Name: ...\n"
                "ID number: ...\n"
                "Expiry: ...\n"
            )

        prompt = (
            "You are analyzing uploaded documents: CVs, plain text, JSON, or "
            "images that were read with local OCR (text may have minor errors).\n\n"
            "Rules:\n"
            "* Extract exact information only from context\n"
            "* If the question is implicit (e.g. \"my name\"), infer correctly from the document\n"
            "* Do not hallucinate; if unsure, say what is unclear\n"
            "* For noisy OCR, prefer quoting short exact phrases from context\n"
            f"{id_instructions}"
            f"Question: {original_question}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )
        logger.info("generation model=%s", LLM_MODEL)
        answer = self.llm.invoke(prompt)
        answer_text = answer.content if hasattr(answer, "content") else str(answer)

        return {
            "answer": answer_text,
            "original_question": original_question,
            "retrieval_query": retrieval_query,
            "query_rewritten": rewritten,
            "retrieved_chunks": [
                {
                    "metadata": doc.metadata,
                    "content": doc.page_content[:1200],
                }
                for doc in docs
            ],
            "k": RETRIEVER_K,
        }
