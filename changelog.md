# Changelog

## 2026-09

- Created `llm-engine` project.
- Added dataset preparation pipeline.
- Added 20 medical QA samples.
- Split into 16 train / 4 validation.
- Loaded Qwen2.5-0.5B-Instruct.
- Tested base model.
- Added LoRA fine-tuning.
- Trained LoRA successfully on CPU.
- Trainable params: 1.08M / 495M (0.2184%).
- Training loss: ~5.27 → ~3.29.
- Validation loss: ~3.75 → ~3.14.
- Saved adapter to `experiments/qwen-medical-lora`.
- Added base vs LoRA evaluation.
- Tested 4 unseen questions.
- Saw some improvement in conciseness, but also some bad/extra answers.
- Added first version of `regression.py`.

## Next

- Run regression evaluation.
- Improve evaluation metrics.
- Get a better/larger dataset.
- Build RAG.

## 2026-09

- Ran `evaluation/regression.py`.
- Base avg latency: 11.666s.
- LoRA avg latency: 17.168s.
- Base keyword coverage: 0.321.
- LoRA keyword coverage: 0.421.
- Saved `experiments/regression_report.json`.

## Next

- Improve dataset/evaluation size.
- Build retrieval/RAG pipeline.

- Replaced Chroma experiment with FAISS after Chroma collection operations hung.
- Built FAISS ingestion pipeline.
- Indexed 6 medical documents.
- Embedding dimension: 384.
- Built semantic retrieval.
- Tested kidney query successfully.
- Added retrieval evaluation.
- Tested 5 queries.
- Recall@1: 1.00.
- Recall@3: 1.00.
- Current retrieval set is only a smoke test, not a production evaluation.
- Added FastAPI inference server.
- Model loaded once at startup.
- Added /health endpoint.
- Added /generate endpoint.
- Verified API returns 200 OK.
- Benchmark: 12.284s latency for 100 output tokens.
- Throughput: 8.14 tokens/sec on CPU.



