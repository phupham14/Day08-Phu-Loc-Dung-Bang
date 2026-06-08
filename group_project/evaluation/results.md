# RAG Evaluation Results

## Framework

- Primary function: `local_fallback`
- Metrics: RAGAS Faithfulness, Answer Relevancy, Context Recall, Context Precision
- Score range: 0.0 to 1.0

## Overall Scores

| Metric | Score |
|--------|------:|
| faithfulness | 0.963 |
| answer_relevance | 0.660 |
| context_recall | 0.825 |
| context_precision | 0.738 |
| average | 0.797 |

## A/B Comparison

| Config | Faithfulness | Relevance | Context Recall | Context Precision | Average |
|--------|-------------:|----------:|---------------:|------------------:|--------:|
| default_top5_reorder | 0.963 | 0.660 | 0.825 | 0.738 | 0.797 |

## Config Notes

- `default_top5_reorder`: Dense retrieval top_k=5 + reorder

## Worst Performers

| # | Question | Average | Likely failure stage |
|---|----------|--------:|----------------------|
| 1 | Nội dung quản lý nhà nước về phòng, chống ma túy bao gồm gì? | 0.715 | answer_relevance |
| 2 | Luật 120/2025/QH15 nghiêm cấm các hành vi nào liên quan đến trồng cây có chứa chất ma túy? | 0.728 | answer_relevance |
| 3 | Nguồn tài chính cho phòng, chống ma túy gồm những nguồn nào? | 0.736 | answer_relevance |

## Recommendations

1. Expand the indexed corpus to include all legal PDFs in `group_project/docs` instead of only the current filtered law document.
2. Add a lexical/BM25 or hybrid retriever for article-number questions, because legal questions often depend on exact phrases like `Điều 55`.
3. Keep RAGAS enabled for final grading and use `--fallback` only for fast offline smoke tests.
