"""
RAG evaluation pipeline for the group project.

Default path:
    python group_project/evaluation/eval_pipeline.py

The default mode uses RAGAS metrics for both the main report and A/B
comparison. The judge model is OpenRouter Nemotron by default.
Use --fallback only when you want a quick offline smoke test.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RAGAS_JUDGE_MODEL = "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

EVAL_DIR = Path(__file__).parent
PROJECT_DIR = EVAL_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
GOLDEN_DATASET_PATH = EVAL_DIR / "golden_dataset.json"
RESULTS_PATH = EVAL_DIR / "results.md"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load_env_files() -> None:
    """Load simple KEY=VALUE pairs from common .env locations."""
    env_paths = [
        Path.cwd() / ".env",
        PROJECT_DIR / ".env",
        PROJECT_DIR.parent / ".env",
    ]
    for env_path in env_paths:
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


_load_env_files()


@dataclass
class PipelineConfig:
    name: str
    description: str
    top_k: int = 5
    use_llm_generation: bool = True
    use_lost_in_middle_reorder: bool = True


class GroupRAGPipeline:
    """Small adapter around retrieval.search() and generation.generate()."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.use_fallback = False
        self.fallback_chunks: list[dict[str, Any]] = []
        try:
            from retrieval import build_vector_db

            build_vector_db()
        except Exception as exc:
            print(f"[WARN] Cannot initialize retrieval.py ({exc}). Using OCR-cache fallback.")
            self.use_fallback = True
            self.fallback_chunks = _load_fallback_chunks()

    def generate_with_citation(self, question: str) -> dict[str, Any]:
        if self.use_fallback:
            chunks = _fallback_search(question, self.fallback_chunks, self.config.top_k)
            return {
                "answer": _extractive_answer(question, chunks),
                "sources": chunks,
                "config": self.config.name,
            }

        from generation import generate, reorder_for_llm
        from retrieval import search

        chunks = search(question, top_k=self.config.top_k)
        llm_chunks = (
            reorder_for_llm(chunks)
            if self.config.use_lost_in_middle_reorder
            else chunks
        )

        answer = ""
        if self.config.use_llm_generation:
            result = generate(question, llm_chunks, history=[])
            answer = result.answer

        if _looks_like_missing_api_answer(answer):
            answer = _extractive_answer(question, chunks)

        return {
            "answer": answer,
            "sources": chunks,
            "config": self.config.name,
        }


def load_golden_dataset() -> list[dict[str, str]]:
    """Load golden dataset from JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if len(data) < 15:
        raise ValueError("golden_dataset.json must contain at least 15 Q&A pairs.")
    return data


def evaluate_with_ragas(
    rag_pipeline: GroupRAGPipeline,
    golden_dataset: list[dict[str, str]],
    *,
    allow_fallback: bool = False,
) -> dict[str, Any]:
    """Evaluate with RAGAS using OpenRouter as the judge LLM provider."""
    if allow_fallback and (
        not os.getenv("OPENROUTER_API_KEY") or os.getenv("RAGAS_DISABLE") == "1"
    ):
        return evaluate_locally(rag_pipeline, golden_dataset, "local_fallback")
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError(
            "OPENROUTER_API_KEY is required for RAGAS judge metrics. "
            "Set it in .env or in your shell before running this script."
        )

    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except Exception as exc:
        if not allow_fallback:
            raise RuntimeError(
                "RAGAS dependencies are missing. Install with: "
                "pip install ragas datasets langchain-openai langchain-core sentence-transformers"
            ) from exc
        return evaluate_locally(rag_pipeline, golden_dataset, "local_fallback")

    judge_model = os.getenv("RAGAS_JUDGE_MODEL", RAGAS_JUDGE_MODEL)
    api_model = _openrouter_api_model(judge_model)
    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=api_model,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
            temperature=0,
        )
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        LocalSentenceTransformerEmbeddings(
            model_name=os.getenv("RAGAS_EMBEDDING_MODEL", "BAAI/bge-m3")
        )
    )

    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    raw_cases = []
    sources_by_index = []

    for index, item in enumerate(golden_dataset, start=1):
        print(f"[ragas:{rag_pipeline.config.name}] {index}/{len(golden_dataset)} {item['question']}")
        result = rag_pipeline.generate_with_citation(item["question"])
        contexts = [chunk.get("content", "") for chunk in result["sources"]]
        eval_data["question"].append(item["question"])
        eval_data["answer"].append(result["answer"])
        eval_data["contexts"].append(contexts)
        eval_data["ground_truth"].append(item["expected_answer"])
        raw_cases.append({
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "answer": result["answer"],
        })
        sources_by_index.append([
            chunk.get("metadata", {}).get("source", chunk.get("source", "unknown"))
            for chunk in result["sources"]
        ])

    dataset = Dataset.from_dict(eval_data)
    ragas_result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )
    df = ragas_result.to_pandas()

    cases = []
    for index, raw_case in enumerate(raw_cases):
        row = df.iloc[index].to_dict()
        scores = _normalize_ragas_scores(row)
        cases.append({
            "question": raw_case["question"],
            "expected_answer": raw_case["expected_answer"],
            "answer": raw_case["answer"],
            "scores": scores,
            "average": statistics.mean(scores.values()),
            "sources": sources_by_index[index],
        })

    return _aggregate_cases(cases, framework=f"ragas:{judge_model}")


def evaluate_with_trulens(
    rag_pipeline: GroupRAGPipeline,
    golden_dataset: list[dict[str, str]],
) -> dict[str, Any]:
    """
    TruLens-compatible entry point.

    TruLens dashboard setup is environment-specific, so this returns the same
    normalized local result shape used by export_results().
    """
    return evaluate_locally(rag_pipeline, golden_dataset, "trulens_compatible_local")


def evaluate_locally(
    rag_pipeline: GroupRAGPipeline,
    golden_dataset: list[dict[str, str]],
    framework: str = "local",
) -> dict[str, Any]:
    """Evaluate RAG outputs with deterministic lexical metrics."""
    cases = []
    for index, item in enumerate(golden_dataset, start=1):
        print(f"[{rag_pipeline.config.name}] {index}/{len(golden_dataset)} {item['question']}")
        result = rag_pipeline.generate_with_citation(item["question"])
        contexts = [chunk.get("content", "") for chunk in result["sources"]]
        cases.append(_score_case(item, result["answer"], contexts, result["sources"]))
    return _aggregate_cases(cases, framework=framework)


def compare_configs(
    rag_pipeline: GroupRAGPipeline | None,
    golden_dataset: list[dict[str, str]],
    *,
    allow_fallback: bool = False,
) -> dict[str, Any]:
    """Compare at least two retrieval/generation configurations."""
    configs = [
        PipelineConfig(
            name="config_a_top5_reorder",
            description="Dense retrieval top_k=5 + lost-in-the-middle reorder before generation",
            top_k=5,
            use_lost_in_middle_reorder=True,
        ),
        PipelineConfig(
            name="config_b_top3_plain",
            description="Dense retrieval top_k=3 without reorder",
            top_k=3,
            use_lost_in_middle_reorder=False,
        ),
    ]

    comparison = {}
    for config in configs:
        pipeline = GroupRAGPipeline(config)
        comparison[config.name] = evaluate_with_ragas(
            pipeline,
            golden_dataset,
            allow_fallback=allow_fallback,
        )
        comparison[config.name]["description"] = config.description
    return comparison


def export_results(results: dict[str, Any], comparison: dict[str, Any]) -> None:
    """Export evaluation results to results.md."""
    lines = [
        "# RAG Evaluation Results",
        "",
        "## Framework",
        "",
        f"- Primary function: `{results['framework']}`",
        "- Metrics: RAGAS Faithfulness, Answer Relevancy, Context Recall, Context Precision",
        "- Score range: 0.0 to 1.0",
        "",
        "## Overall Scores",
        "",
        "| Metric | Score |",
        "|--------|------:|",
    ]
    for metric, score in results["overall"].items():
        lines.append(f"| {metric} | {score:.3f} |")

    lines.extend([
        "",
        "## A/B Comparison",
        "",
        "| Config | Faithfulness | Relevance | Context Recall | Context Precision | Average |",
        "|--------|-------------:|----------:|---------------:|------------------:|--------:|",
    ])
    for name, report in comparison.items():
        overall = report["overall"]
        lines.append(
            f"| {name} | {overall['faithfulness']:.3f} | "
            f"{overall['answer_relevance']:.3f} | {overall['context_recall']:.3f} | "
            f"{overall['context_precision']:.3f} | {overall['average']:.3f} |"
        )

    lines.extend(["", "## Config Notes", ""])
    for name, report in comparison.items():
        lines.append(f"- `{name}`: {report.get('description', '')}")

    lines.extend([
        "",
        "## Worst Performers",
        "",
        "| # | Question | Average | Likely failure stage |",
        "|---|----------|--------:|----------------------|",
    ])
    worst_cases = sorted(results["cases"], key=lambda item: item["average"])[:3]
    for idx, case in enumerate(worst_cases, start=1):
        question = case["question"].replace("|", " ")
        stage = _failure_stage(case)
        lines.append(f"| {idx} | {question} | {case['average']:.3f} | {stage} |")

    lines.extend([
        "",
        "## Recommendations",
        "",
        "1. Expand the indexed corpus to include all legal PDFs in `group_project/docs` instead of only the current filtered law document.",
        "2. Add a lexical/BM25 or hybrid retriever for article-number questions, because legal questions often depend on exact phrases like `Điều 55`.",
        "3. Keep RAGAS enabled for final grading and use `--fallback` only for fast offline smoke tests.",
        "",
    ])

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")


def _load_fallback_chunks(chunk_size: int = 1200, overlap: int = 200) -> list[dict[str, Any]]:
    cache_dir = PROJECT_DIR / "ocr_cache"
    cache_files = sorted(cache_dir.glob("*.txt"))
    chunks: list[dict[str, Any]] = []
    for cache_file in cache_files:
        text = cache_file.read_text(encoding="utf-8", errors="ignore")
        step = max(1, chunk_size - overlap)
        for index, start in enumerate(range(0, len(text), step)):
            content = text[start:start + chunk_size].strip()
            if not content:
                continue
            chunks.append({
                "content": content,
                "score": 0.0,
                "source": cache_file.stem,
                "metadata": {
                    "source": cache_file.stem,
                    "type": "ocr_cache",
                    "chunk_index": index,
                },
            })
    if not chunks:
        raise FileNotFoundError(f"No OCR cache text files found in {cache_dir}")
    return chunks


def _fallback_search(
    question: str,
    chunks: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    query_tokens = _tokens(question)
    scored = []
    for chunk in chunks:
        chunk_tokens = _tokens(chunk.get("content", ""))
        if not chunk_tokens:
            score = 0.0
        else:
            score = len(query_tokens & chunk_tokens) / len(query_tokens or {"_none"})
        candidate = {**chunk, "score": round(score, 4)}
        scored.append(candidate)
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def _score_case(
    item: dict[str, str],
    answer: str,
    contexts: list[str],
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    context_text = "\n".join(contexts)
    expected = item["expected_answer"]
    expected_context = item.get("expected_context", "")

    faithfulness = _coverage(answer, context_text)
    answer_relevance = statistics.mean([
        _coverage(item["question"], answer),
        _coverage(expected, answer),
    ])
    context_recall = statistics.mean([
        _coverage(expected, context_text),
        _coverage(expected_context, context_text),
    ])
    precision_scores = [
        statistics.mean([
            _coverage(item["question"], chunk.get("content", "")),
            _coverage(expected, chunk.get("content", "")),
        ])
        for chunk in sources
    ]
    context_precision = statistics.mean(precision_scores) if precision_scores else 0.0

    scores = {
        "faithfulness": faithfulness,
        "answer_relevance": answer_relevance,
        "context_recall": context_recall,
        "context_precision": context_precision,
    }
    return {
        "question": item["question"],
        "expected_answer": expected,
        "answer": answer,
        "scores": scores,
        "average": statistics.mean(scores.values()),
        "sources": [
            chunk.get("metadata", {}).get("source", chunk.get("source", "unknown"))
            for chunk in sources
        ],
    }


def _aggregate_cases(cases: list[dict[str, Any]], framework: str) -> dict[str, Any]:
    metrics = ["faithfulness", "answer_relevance", "context_recall", "context_precision"]
    overall = {
        metric: statistics.mean(case["scores"][metric] for case in cases)
        for metric in metrics
    }
    overall["average"] = statistics.mean(overall.values())
    return {"framework": framework, "overall": overall, "cases": cases}


class LocalSentenceTransformerEmbeddings:
    """Minimal LangChain-compatible embeddings wrapper for RAGAS."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _openrouter_api_model(judge_model: str) -> str:
    """RAGAS report keeps openrouter/... while OpenRouter API expects provider/model."""
    prefix = "openrouter/"
    if judge_model.startswith(prefix):
        return judge_model[len(prefix):]
    return judge_model


def _normalize_ragas_scores(row: dict[str, Any]) -> dict[str, float]:
    return {
        "faithfulness": _as_float(row.get("faithfulness")),
        "answer_relevance": _as_float(
            row.get("answer_relevancy", row.get("answer_relevance"))
        ),
        "context_recall": _as_float(row.get("context_recall")),
        "context_precision": _as_float(row.get("context_precision")),
    }


def _as_float(value: Any) -> float:
    try:
        if value is None or value != value:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _coverage(needle: str, haystack: str) -> float:
    needle_tokens = _tokens(needle)
    haystack_tokens = _tokens(haystack)
    if not needle_tokens:
        return 0.0
    return len(needle_tokens & haystack_tokens) / len(needle_tokens)


def _tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFD", text.lower())
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if len(token) > 1 and token not in _STOPWORDS
    }


def _extractive_answer(question: str, chunks: list[dict[str, Any]], max_chars: int = 900) -> str:
    if not chunks:
        return "Không tìm thấy ngữ cảnh phù hợp trong kho tài liệu hiện có."
    snippets = []
    for chunk in chunks[:3]:
        content = " ".join(chunk.get("content", "").split())
        snippets.append(content[:300])
    source = chunks[0].get("metadata", {}).get("source", chunks[0].get("source", "unknown"))
    answer = " ".join(snippets)
    return f"{answer[:max_chars]} [{source}]"


def _looks_like_missing_api_answer(answer: str) -> bool:
    if not answer:
        return True
    lowered = answer.lower()
    markers = ["groq_api_key", "openai_api_key", "chưa được cấu hình", "vui lòng thêm"]
    return any(marker in lowered for marker in markers)


def _failure_stage(case: dict[str, Any]) -> str:
    scores = case["scores"]
    if scores["context_recall"] < 0.45:
        return "retrieval/context_recall"
    if scores["context_precision"] < 0.35:
        return "retrieval/context_precision"
    if scores["faithfulness"] < 0.45:
        return "generation/faithfulness"
    return "answer_relevance"


def _short_reason(case: dict[str, Any], max_chars: int = 180) -> str:
    reasons = case.get("reasons", {})
    if not reasons:
        return "N/A"
    weakest_metric = min(case["scores"], key=case["scores"].get)
    reason = " ".join(str(reasons.get(weakest_metric, "")).split())
    if not reason:
        return "N/A"
    return reason[:max_chars] + ("..." if len(reason) > max_chars else "")


_STOPWORDS = {
    "la", "va", "cua", "co", "cac", "cho", "trong", "theo", "duoc", "ve",
    "nhung", "nao", "gi", "mot", "voi", "nguoi", "quy", "dinh", "luat",
    "phong", "chong", "ma", "tuy", "qhid", "qh15", "dieu", "khoan",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the group RAG pipeline.")
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Use deterministic local metrics if RAGAS/OpenRouter setup is unavailable.",
    )
    parser.add_argument(
        "--no-ab",
        action="store_true",
        help="Skip A/B comparison to reduce RAGAS/OpenRouter API calls.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks for the default config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    default_config = PipelineConfig(
        name="default_top5_reorder",
        description="Dense retrieval top_k=5 + reorder",
        top_k=args.top_k,
        use_lost_in_middle_reorder=True,
    )
    pipeline = GroupRAGPipeline(default_config)
    results = evaluate_with_ragas(
        pipeline,
        golden_dataset,
        allow_fallback=args.fallback,
    )
    if args.no_ab:
        comparison = {
            default_config.name: {
                **results,
                "description": default_config.description,
            }
        }
    else:
        comparison = compare_configs(
            pipeline,
            golden_dataset,
            allow_fallback=args.fallback,
        )
    export_results(results, comparison)
    print(f"Saved report to {RESULTS_PATH}")
