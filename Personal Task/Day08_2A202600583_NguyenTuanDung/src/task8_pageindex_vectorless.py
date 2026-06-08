"""
Task 8 - PageIndex Vectorless RAG.

PageIndex supports document retrieval without building our own vector store. In
this module, documents are uploaded to PageIndex, then queries are sent to the
PageIndex retrieval API.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STANDARDIZED_DIR = PROJECT_ROOT / "data" / "standardized"
LANDING_LEGAL_DIR = PROJECT_ROOT / "data" / "landing" / "legal"
PAGEINDEX_DOCS_FILE = PROJECT_ROOT / "data" / "pageindex_documents.json"

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")


def _get_pageindex_client():
    if not PAGEINDEX_API_KEY or PAGEINDEX_API_KEY == "pi_xxx":
        raise ValueError("Missing PAGEINDEX_API_KEY in .env")

    from pageindex import PageIndexClient

    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _load_uploaded_doc_ids() -> list[str]:
    if not PAGEINDEX_DOCS_FILE.exists():
        return []

    data = json.loads(PAGEINDEX_DOCS_FILE.read_text(encoding="utf-8"))
    return [item["doc_id"] for item in data.get("documents", []) if item.get("doc_id")]


def _save_uploaded_docs(documents: list[dict[str, Any]]) -> None:
    PAGEINDEX_DOCS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PAGEINDEX_DOCS_FILE.write_text(
        json.dumps({"documents": documents}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _candidate_files() -> list[Path]:
    """
    Prefer original legal PDFs because the installed PageIndex SDK uploads files.

    Markdown files are still used by the local fallback search below.
    """
    pdf_files = sorted(LANDING_LEGAL_DIR.rglob("*.pdf")) if LANDING_LEGAL_DIR.exists() else []
    if pdf_files:
        return pdf_files
    return sorted(STANDARDIZED_DIR.rglob("*.md"))


def upload_documents() -> list[dict[str, Any]]:
    """
    Upload available documents to PageIndex and persist returned doc_ids locally.

    Returns:
        List of {'filename': str, 'doc_id': str, 'type': str}.
    """
    client = _get_pageindex_client()
    files = _candidate_files()
    if not files:
        raise FileNotFoundError("No documents found to upload")

    uploaded_docs: list[dict[str, Any]] = []
    for file_path in files:
        try:
            result = client.submit_document(str(file_path))
        except Exception as exc:
            print(f"  Skipped {file_path.name}: {exc}")
            continue

        doc_id = result.get("doc_id") or result.get("id")
        if not doc_id:
            print(f"  Skipped {file_path.name}: PageIndex response has no doc_id")
            continue

        uploaded_doc = {
            "filename": file_path.name,
            "doc_id": doc_id,
            "type": file_path.parent.name,
        }
        uploaded_docs.append(uploaded_doc)
        print(f"  Uploaded: {file_path.name} -> {doc_id}")

    if not uploaded_docs:
        raise RuntimeError("No documents were uploaded to PageIndex")

    _save_uploaded_docs(uploaded_docs)
    return uploaded_docs


def _extract_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return str(item)

    for key in ("content", "text", "markdown", "answer", "snippet"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value

    return json.dumps(item, ensure_ascii=False)


def _extract_score(item: Any, default_score: float) -> float:
    if isinstance(item, dict):
        for key in ("score", "relevance_score", "confidence"):
            value = item.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return default_score


def _extract_metadata(item: Any, doc_id: str) -> dict[str, Any]:
    metadata = {"doc_id": doc_id}
    if isinstance(item, dict):
        raw_metadata = item.get("metadata")
        if isinstance(raw_metadata, dict):
            metadata.update(raw_metadata)
        for key in ("page", "page_num", "filename", "node_id"):
            if key in item:
                metadata[key] = item[key]
    return metadata


def _normalize_retrieval_response(
    response: dict[str, Any],
    doc_id: str,
    top_k: int,
) -> list[dict[str, Any]]:
    raw_items = (
        response.get("results")
        or response.get("chunks")
        or response.get("retrieval")
        or response.get("references")
        or []
    )

    if isinstance(raw_items, dict):
        raw_items = raw_items.get("results") or raw_items.get("chunks") or [raw_items]
    if not isinstance(raw_items, list):
        raw_items = [raw_items]

    results = []
    for rank, item in enumerate(raw_items[:top_k], start=1):
        results.append(
            {
                "content": _extract_text(item),
                "score": _extract_score(item, 1.0 / rank),
                "metadata": _extract_metadata(item, doc_id),
                "source": "pageindex",
            }
        )
    return results


def _query_doc(
    client: Any,
    doc_id: str,
    query: str,
    top_k: int,
    max_wait_seconds: int = 10,
) -> list[dict[str, Any]]:
    submitted = client.submit_query(doc_id=doc_id, query=query, thinking=False)
    retrieval_id = submitted.get("retrieval_id") or submitted.get("id")
    if not retrieval_id:
        return _normalize_retrieval_response(submitted, doc_id, top_k)

    deadline = time.time() + max_wait_seconds
    latest_response: dict[str, Any] = {}
    while time.time() < deadline:
        latest_response = client.get_retrieval(retrieval_id)
        status = str(latest_response.get("status", "")).lower()
        if status in {"completed", "complete", "ready", "success", "succeeded", "done"}:
            return _normalize_retrieval_response(latest_response, doc_id, top_k)
        if status in {"failed", "error"}:
            raise RuntimeError(f"PageIndex retrieval failed: {latest_response}")
        time.sleep(2)

    return _normalize_retrieval_response(latest_response, doc_id, top_k)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _local_markdown_fallback(query: str, top_k: int) -> list[dict[str, Any]]:
    """
    Small vectorless fallback over standardized markdown files.

    This keeps Task 8 runnable when PageIndex API credentials are not configured.
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    results: list[dict[str, Any]] = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
        for i, paragraph in enumerate(paragraphs):
            paragraph_tokens = _tokenize(paragraph)
            overlap = len(query_tokens & paragraph_tokens)
            if overlap == 0:
                continue
            score = overlap / len(query_tokens)
            results.append(
                {
                    "content": paragraph,
                    "score": score,
                    "metadata": {
                        "filename": md_file.name,
                        "type": md_file.parent.name,
                        "chunk_index": i,
                        "fallback": True,
                    },
                    "source": "pageindex",
                }
            )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval using PageIndex.

    Args:
        query: User query.
        top_k: Maximum number of results.

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'
        }
    """
    if top_k <= 0:
        return []

    try:
        client = _get_pageindex_client()
        doc_ids = _load_uploaded_doc_ids()

        if not doc_ids:
            raise ValueError("No cached PageIndex doc_ids found. Run upload_documents() first.")

        results: list[dict[str, Any]] = []
        for doc_id in doc_ids:
            results.extend(_query_doc(client, doc_id, query, top_k))

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]
    except Exception as exc:
        print(f"PageIndex unavailable ({exc}). Using local markdown fallback.")
        return _local_markdown_fallback(query, top_k)


if __name__ == "__main__":
    import sys

    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    print("Task 8: PageIndex Vectorless RAG")

    if PAGEINDEX_API_KEY and PAGEINDEX_API_KEY != "pi_xxx":
        print("Uploading documents...")
        try:
            upload_documents()
        except Exception as exc:
            print(f"Upload skipped: {exc}")
    else:
        print("PAGEINDEX_API_KEY is not configured. Skipping upload.")

    print("\nTest query:")
    results = pageindex_search("hinh phat su dung ma tuy", top_k=3)
    for result in results:
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
