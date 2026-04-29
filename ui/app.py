import os

import requests
import streamlit as st

# Default 8010 — port 8000 is often another web app (HTML 404 instead of JSON /health).
API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8010")

st.set_page_config(page_title="Local RAG UI", layout="wide")
st.title("Local RAG Demo")

st.subheader("System Status")
if st.button("Check Health"):
    try:
        response = requests.get(f"{API_BASE}/health", timeout=20)
        if response.ok:
            st.success("Backend is healthy.")
            st.json(response.json())
        else:
            st.error(f"Health check failed: {response.text}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Cannot reach backend: {exc}")

st.subheader("Upload Files (PDF / TXT / JSON / Images for OCR)")
uploaded_files = st.file_uploader(
    "Select files",
    type=["pdf", "txt", "json", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if st.button("Upload and Index"):
    if not uploaded_files:
        st.warning("Please upload at least one file.")
    else:
        files_payload = [
            ("files", (f.name, f.getvalue(), "application/octet-stream"))
            for f in uploaded_files
        ]
        try:
            with st.spinner("Uploading and indexing..."):
                response = requests.post(
                    f"{API_BASE}/upload", files=files_payload, timeout=300
                )
            if response.ok:
                st.success("Files indexed.")
                st.json(response.json())
            else:
                st.error(response.text)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Upload failed: {exc}")

st.subheader("Ask Question")
question = st.text_input("Question", placeholder="Ask about your uploaded files...")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please type a question.")
    else:
        try:
            with st.spinner("Generating answer..."):
                response = requests.post(
                    f"{API_BASE}/ask",
                    json={"question": question},
                    timeout=180,
                )
            if response.ok:
                result = response.json()
                st.markdown("### Answer")
                st.write(result["answer"])
                if result.get("query_rewritten"):
                    with st.expander("Debug: retrieval query rewrite"):
                        st.write("Original:", result.get("original_question"))
                        st.write("Retrieval:", result.get("retrieval_query"))
                st.markdown(f"### Retrieved Chunks (k={result['k']})")
                for idx, chunk in enumerate(result["retrieved_chunks"], start=1):
                    with st.expander(f"Chunk {idx}"):
                        st.write(chunk["metadata"])
                        st.code(chunk["content"])
            else:
                st.error(response.text)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Ask failed: {exc}")
