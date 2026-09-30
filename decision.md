# Decisions

## 2026-09

- Goal: build a small project matching the important parts of the Recollia role.
- Local machine is CPU-only with 16 GB RAM, so use a small model.
- Started with Qwen2.5-0.5B-Instruct.
- Use LoRA instead of full fine-tuning.
- First dataset is only a smoke test (20 medical QA samples).
- Planned path: fine-tuning → evaluation → RAG → inference → model gateway.

## 2026-09

- First regression run completed.
- LoRA keyword coverage was higher (0.421 vs 0.321).
- LoRA was slower (17.17s vs 11.67s).
- Only 4 validation samples, so no strong conclusion yet.
- Need a larger evaluation set before claiming improvement.
- Next major feature: RAG/retrieval.
- Chroma was hanging during collection operations on this setup.
- Switched to FAISS for the first RAG implementation.
- Used Recall@K to evaluate retrieval instead of judging results manually.
- Kept the retrieval dataset small for now and improve it later.
- Load the model once when the API starts instead of per request.
- Keep raw model generation and RAG generation as separate endpoints.
- Measure latency and tokens/sec instead of only checking output quality.