"""Manual validation script for hybrid Document AI endpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests


def upload_file(api_base: str, file_path: Path) -> dict:
    with file_path.open("rb") as fh:
        response = requests.post(
            f"{api_base}/upload",
            files={"files": (file_path.name, fh, "application/octet-stream")},
            timeout=240,
        )
    response.raise_for_status()
    return response.json()


def ask_question(api_base: str, question: str) -> dict:
    response = requests.post(
        f"{api_base}/ask",
        json={"question": question},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate hybrid Document AI behavior.")
    parser.add_argument("--api-base", default="http://127.0.0.1:8010")
    parser.add_argument("--cv-pdf", type=Path, required=True)
    parser.add_argument("--id-image", type=Path, required=True)
    parser.add_argument("--invoice-file", type=Path, required=True)
    args = parser.parse_args()

    scenarios = [
        (args.cv_pdf, "what is my name"),
        (args.id_image, "what is my id number"),
        (args.invoice_file, "what is total amount"),
    ]

    for file_path, question in scenarios:
        print(f"\n=== Scenario: {file_path.name} ===")
        upload_result = upload_file(args.api_base, file_path)
        print("upload:", upload_result.get("processed_documents", []))
        answer_result = ask_question(args.api_base, question)
        print("mode:", answer_result.get("mode"))
        print("doc_type:", answer_result.get("document_type"))
        print("structured_data:", answer_result.get("structured_data"))
        print("answer:", answer_result.get("answer"))


if __name__ == "__main__":
    main()
