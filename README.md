# RAG: Baseline Bi-Encoder Retrieval vs. Cross-Encoder Re-Ranking

Two retrieval pipelines are compared over the same chunked corpus, the GDPR (Regulation (EU) 2016/679, 88 pages):

- **Pipeline 1:** `all-MiniLM-L6-v2` bi-encoder → FAISS cosine Top-10 → top 3 go to the LLM
- **Pipeline 2:** the same Top-10 → `cross-encoder/ms-marco-MiniLM-L-6-v2` re-rank → top 3 go to the LLM

LLM: local `Qwen/Qwen2.5-3B-Instruct`, or Gemini if `GEMINI_API_KEY` is set.

| File | Contents |
|---|---|
| `rag_rerank.ipynb` | Executed notebook: chunking, both pipelines, failure queries, rank shifts, answers, latency, scaling |
| `RAG_Report.docx` | Written report |
| `build_notebook.py` | Generates the notebook source |
| `data/gdpr.pdf` | Source document (downloaded automatically if missing) |

## Run
```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 rag_rerank.ipynb
```

## Key result
| Query | Source | Bi-encoder rank | Cross-encoder rank |
|---|---|---|---|
| Max fine for basic principles / consent | Art. 83(5) | #4 | #1 |
| When does the right to erasure not apply? | Art. 17(3) | #4 | #2 |
| Is profiling permitted? | Art. 22 | #5 | #2 |
| Processor liability | Art. 82(2) | #5 | #1 |
| When can processing be restricted? | Art. 18(1) | #5 | #2 |
| Transfers for public interest | Recital 112 | #8 | #2 |

Re-ranking adds about 50 ms per query, about 0.5% of end-to-end latency.
