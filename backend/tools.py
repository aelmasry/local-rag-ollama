import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger


def retry(
    attempts: int = 3, delay_seconds: float = 1.0
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error: Exception | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    if attempt < attempts:
                        time.sleep(delay_seconds * attempt)
            if last_error is None:
                raise RuntimeError("Operation failed without a captured exception.")
            raise last_error

        return wrapper

    return decorator


def flatten_json(value: Any, prefix: str = "") -> list[str]:
    lines: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            lines.extend(flatten_json(item, next_prefix))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            next_prefix = f"{prefix}[{idx}]"
            lines.extend(flatten_json(item, next_prefix))
    else:
        lines.append(f"{prefix}: {'' if value is None else value}")
    return lines


def json_file_to_documents(path: Path) -> list[Document]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        docs = []
        for idx, item in enumerate(data):
            docs.append(
                Document(
                    page_content="\n".join(flatten_json(item)),
                    metadata={"source": path.name, "record_index": idx},
                )
            )
        return docs

    if isinstance(data, dict):
        return [
            Document(
                page_content="\n".join(flatten_json(data)),
                metadata={"source": path.name, "record_index": 0},
            )
        ]

    return [
        Document(
            page_content=str(data), metadata={"source": path.name, "record_index": 0}
        )
    ]
