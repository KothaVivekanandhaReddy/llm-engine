# LLM Engine

A modular LLM engineering stack for experimenting with **fine-tuning, evaluation, RAG, inference serving, model routing, testing, and containerized deployment**.

The project explores the engineering path from **domain-specific data → model adaptation → evaluation → retrieval → RAG → inference gateway → API → Docker**.

> **Project status:** Engineering / learning prototype  
> **Domain:** Medical education  
> **Primary constraint:** CPU-based local development  
>
> This project is not intended for clinical diagnosis or medical decision-making.

---

## Architecture

```text
                         ┌──────────────────────┐
                         │       FastAPI        │
                         │    HTTP Interface    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     Model Router     │
                         │ routing + fallback   │
                         └───────┬────────┬──────┘
                                 │        │
                      ┌──────────▼──┐   ┌─▼──────────┐
                      │ Local Qwen  │   │  Local RAG  │
                      │  Provider   │   │  Provider   │
                      └─────────────┘   └──────┬──────┘
                                               │
                                  ┌────────────▼───────────┐
                                  │ Sentence Transformers  │
                                  │          + FAISS        │
                                  └─────────────────────────┘
```

### Training / Evaluation

```text
Medical QA
    │
    ▼
Data Preparation
    │
    ▼
LoRA / PEFT
    │
    ▼
Fine-tuned Adapter
    │
    ▼
Held-out Evaluation
    │
    ▼
Regression Testing
```

### RAG

```text
User Question
      │
      ▼
Query Embedding
      │
      ▼
FAISS Similarity Search
      │
      ▼
Top-K Documents
      │
      ▼
Context Construction
      │
      ▼
Qwen Generation
      │
      ▼
Answer + Retrieval Metadata
```

---

## What I Built

This project focuses on the engineering layers around an LLM rather than only model training.

### Model adaptation
- Qwen2.5-0.5B-Instruct
- LoRA / PEFT fine-tuning
- Domain-specific medical QA data
- Train / validation / held-out test split
- Parameter-efficient training on CPU-constrained hardware

### Evaluation
- Held-out QA evaluation
- Keyword-coverage measurement
- Regression evaluation
- Retrieval Recall@K
- Latency measurement
- Retrieval vs. generation profiling

### Retrieval & RAG
- Sentence Transformers embeddings
- Normalized embeddings
- FAISS vector search
- Reusable retriever abstraction
- Top-K retrieval
- RAG provider
- Retrieval metadata in API responses

### Inference infrastructure
- Local Qwen provider
- RAG provider
- Provider abstraction
- Model router
- Failure / fallback provider
- FastAPI inference service
- Dockerized runtime

### Software engineering
- Typed request / response models
- Unit tests
- API tests
- Retrieval tests
- Gateway tests
- Integration checks
- Separate development and runtime dependencies

---

## Technical Stack

| Area | Technology |
|---|---|
| Language | Python |
| Base Model | Qwen2.5-0.5B-Instruct |
| Fine-tuning | Transformers, PEFT, LoRA |
| Dataset | Custom medical QA dataset |
| Embeddings | Sentence Transformers |
| Vector Search | FAISS |
| Inference | Hugging Face Transformers |
| API | FastAPI + Uvicorn |
| Gateway | Custom provider / router abstraction |
| Testing | Pytest |
| Containerization | Docker |
| Runtime | CPU-compatible local deployment |

---

## Dataset

The project uses a small domain-specific medical QA dataset for engineering experimentation.

Current dataset:

- **145 unique QA records**
- **116 training examples**
- **14 validation examples**
- **15 held-out test examples**

The held-out test set is kept separate from training and validation to measure behavior on unseen questions.

Example topics include:

- Chromosomes
- Protein synthesis
- Hemoglobin
- Heart rate
- Cardiac output
- Gas exchange
- Resting membrane potential
- Diabetes mellitus
- Atherosclerosis
- Kidney function
- Tubular reabsorption

The dataset is intentionally small. The goal is to demonstrate the **LLM engineering pipeline**, not production-scale medical domain adaptation.

---

# Fine-Tuning

The base model is:

```text
Qwen2.5-0.5B-Instruct
          │
          ▼
    Medical QA Data
          │
          ▼
   Data Preparation
          │
          ▼
       LoRA / PEFT
          │
          ▼
   Fine-tuned Adapter
          │
          ▼
   Held-out Evaluation
```

### Why LoRA?

LoRA adapts the model by training a relatively small number of additional parameters while keeping most of the base model frozen.

This was useful for the project's hardware constraints because it provides:

- Lower memory requirements
- Faster experimentation
- Smaller trainable parameter count
- Easier adapter management
- Practical CPU-constrained experimentation

### Training statistics

```text
Trainable parameters: 1,081,344
Total parameters:      495,114,112
Trainable percentage:  ~0.2184%
```

---

# Fine-Tuning Results

One of the useful findings from the experiment was that **lower latency did not automatically mean better task performance**.

| Metric | Base | LoRA |
|---|---:|---:|
| Average latency | ~12.41 s | ~2.14 s |
| Average answer length | ~507.5 | ~94.8 |
| Keyword coverage | ~0.61 | ~0.448 |

The LoRA model produced substantially shorter outputs and lower observed latency, but keyword coverage decreased on the current evaluation set.

This illustrates an important engineering principle:

> **Model adaptation should be evaluated on task-specific behavior, not only training loss or inference speed.**

---

# Retrieval-Augmented Generation

The RAG pipeline uses:

```text
Question
   │
   ▼
Sentence Transformer
   │
   ▼
Query Embedding
   │
   ▼
FAISS
   │
   ▼
Top-K Documents
   │
   ▼
Context
   │
   ▼
Qwen
   │
   ▼
Generated Answer
```

### Embedding model

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embeddings are normalized before similarity search.

### Retrieval backend

```text
FAISS
```

The retrieval implementation is separated into a reusable `Retriever` abstraction so that retrieval can be shared across:

- RAG inference
- Retrieval evaluation
- API endpoints
- Future agent workflows

---

# Retrieval Evaluation

Retrieval quality is evaluated independently from final answer generation.

Current experiment:

```text
Questions evaluated: 5

Recall@1: 1.00
Recall@3: 1.00
```

Example:

```text
Question:
What does the kidney do?

Top result:
Kidney Function

Score:
0.7205
```

The current FAISS experiment indexes a small retrieval corpus. This should not be interpreted as the complete 145-record QA dataset being represented by only six vectors; the index is an experimental retrieval subset.

---

# Inference Gateway

The project separates **model selection from the HTTP API** using a provider abstraction.

```text
                  ModelRouter
                       │
             ┌─────────┼─────────┐
             │         │         │
             ▼         ▼         ▼
        Local Qwen  Local RAG  Mock Failure
         Provider    Provider    Provider
```

A common request / response interface allows providers to be replaced without tightly coupling the API to a specific model implementation.

Example request:

```python
GenerateRequest(
    prompt="What is hemoglobin?",
    model="local-qwen",
    max_tokens=100,
    temperature=0.0
)
```

The response can expose:

```text
text
model
provider
latency_seconds
output_tokens
cost
metadata
```

---

# Provider Routing & Fallbacks

Current providers:

### `local-qwen`

Direct local inference using Qwen2.5-0.5B-Instruct.

### `local-rag`

Retrieves relevant documents using FAISS and generates an answer using the retrieved context.

### `mock-failure`

A deliberately failing provider used to test fallback behavior.

```text
Request
   │
   ▼
Selected Provider
   │
   ├── Success ───────► Response
   │
   └── Failure
          │
          ▼
       Fallback
          │
          ▼
       Response
```

This provides the foundation for a future provider-agnostic gateway supporting hosted and open-weight model endpoints.

---

# Performance

The benchmark separates retrieval latency from generation latency.

Example Docker RAG request:

```text
Total latency:       ~4.60 s
Retrieval latency:   ~0.035 s
Generation latency:  ~4.56 s
Generation speed:    ~8.77 tokens/s
```

The experiment shows that, in the current CPU configuration, **generation dominates total latency**, while retrieval contributes very little.

This provides a concrete optimization direction rather than optimizing components based on assumptions.

---

# FastAPI Service

Available endpoints:

```text
GET  /health
POST /generate
POST /rag
POST /gateway/generate
```

### Health

```http
GET /health
```

Example:

```json
{
  "status": "ok",
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "device": "cpu",
  "retrieval": "faiss"
}
```

### Direct generation

```http
POST /generate
```

```json
{
  "question": "What is hemoglobin?",
  "max_new_tokens": 100
}
```

### RAG generation

```http
POST /rag
```

```json
{
  "question": "What does the kidney do?",
  "max_new_tokens": 100
}
```

### Gateway generation

```http
POST /gateway/generate
```

```json
{
  "prompt": "Explain cardiac output.",
  "model": "local-qwen",
  "max_tokens": 100,
  "temperature": 0.0
}
```

FastAPI also exposes interactive Swagger documentation at:

```text
http://localhost:8000/docs
```

---

# Testing

The project uses multiple validation layers:

```text
                 Testing Strategy
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
    Unit Tests      API Tests    Retrieval Eval
        │              │              │
     Gateway       FastAPI        Recall@K
     Retrieval     endpoints
        │
        └──────────────┬──────────────┘
                       ▼
                Model Evaluation
                       │
                       ▼
                Held-out QA Set
                       │
                       ▼
              Integration Checks
```

Current automated test suite:

```text
11 tests
11 passed
```


---

# Docker

The runtime can be packaged as a Docker image.

Build:

```bash
docker build -t llm-engine:0.1 .
```

Run:

```bash
docker run --rm -p 8000:8000 llm-engine:0.1
```

Then:

```text
http://localhost:8000/docs
```

The runtime image contains the components required for:

```text
FastAPI
   +
Gateway
   +
Qwen
   +
FAISS
```

Development-only dependencies and training artifacts are excluded from the runtime environment.

---

# Project Structure

```text
llm-engine/
│
├── configs/
├── data/
│   ├── raw/
│   └── processed/
│
├── finetuning/
│   ├── prepare.py
│   ├── train_lora.py
│   └── train_sft.py
│
├── inference/
│   ├── baseline.py
│   ├── benchmark.py
│   ├── quantize.py
│   ├── rag.py
│   └── server.py
│
├── retrieval/
│   ├── evaluate.py
│   ├── ingest.py
│   └── retrieve.py
│
├── gateway/
│   ├── models.py
│   ├── providers.py
│   ├── router.py
│   ├── mock_provider.py
│   └── rag_provider.py
│
├── evaluation/
│   ├── evaluate.py
│   └── regression.py
│
├── tests/
│   ├── test_api.py
│   ├── test_gateway.py
│   ├── test_retrieval.py
│   └── gateway_check.py
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── requirements-runtime.txt
└── README.md
```

---

# Engineering Decisions

## Why FAISS?

FAISS provides lightweight local vector search and is sufficient for the current experimental corpus.

The project previously experimented with ChromaDB; FAISS became the canonical retrieval backend for this version.

## Why a provider abstraction?

The API should not depend directly on one model implementation.

```text
API
 ↓
Router
 ↓
Provider
 ↓
Model
```

This makes future provider integrations possible without redesigning the API layer.

Potential future providers include:

- OpenAI
- Anthropic
- vLLM
- SGLang
- llama.cpp
- Other open-weight models

## Why evaluate retrieval independently?

RAG quality depends on retrieval quality.

Therefore retrieval is measured independently using Recall@K rather than relying only on final generated answers.

## Why separate retrieval and generation latency?

Total RAG latency can be decomposed into:

```text
Total latency
     =
retrieval
     +
context construction
     +
generation
```

This makes it possible to identify the actual system bottleneck.

---

# Current Limitations

This is an engineering prototype, not a production LLM platform.

### Model

The current model is:

```text
Qwen2.5-0.5B-Instruct
```

It is intentionally small enough for CPU experimentation.

### Dataset

The current dataset contains only 145 examples and is not sufficient for production-grade medical domain adaptation.

### Hardware

Development and inference are currently CPU-based.

### Retrieval

The current implementation uses dense FAISS retrieval.

Production-oriented retrieval would require additional work such as:

- Hybrid BM25 + dense retrieval
- Reranking
- Metadata filtering
- Better chunking
- Larger retrieval evaluation sets
- Retrieval failure analysis

### Gateway

The current gateway demonstrates abstraction and fallback behavior, but is not yet a production multi-provider gateway.

Missing production features include:

- Real multi-provider routing
- API-key management
- Rate limiting
- Streaming
- Provider health checks
- Retry policies
- Distributed tracing
- Production cost accounting

### Quantization

A quantization experiment exists, but native llama.cpp quantization was not completed because of the current Windows/MSVC environment.

---

# Key Engineering Lessons

### 1. Model quality and system quality are different

A model can become faster or have lower training loss without becoming better for the target task.

### 2. Latency must be decomposed

Profiling retrieval and generation independently revealed that generation is the dominant bottleneck in the current CPU configuration.

### 3. Retrieval should be evaluated independently

Recall@K provides a measurable signal for whether the retrieval system is finding the intended evidence.

### 4. Interfaces make models replaceable

A provider abstraction prevents the application layer from being tightly coupled to one inference implementation.

### 5. Deployment should be reproducible

Docker packages the runtime environment and provides a repeatable way to run the inference API.

---

# End-to-End Flow

```text
                 Domain Data
                      │
                      ▼
                Fine-Tuning
                      │
                      ▼
                 Evaluation
                      │
                      ▼
                  Embeddings
                      │
                      ▼
                  Retrieval
                      │
                      ▼
                     RAG
                      │
                      ▼
                 Inference
                      │
                      ▼
                  Gateway
                      │
                      ▼
                    API
                      │
                      ▼
                   Tests
                      │
                      ▼
                   Docker
```

## Purpose

The purpose of this project is to demonstrate practical LLM engineering across the stack:

```text
Data
 ↓
Fine-Tuning
 ↓
Evaluation
 ↓
Embeddings
 ↓
Retrieval
 ↓
RAG
 ↓
Inference
 ↓
Gateway
 ↓
API
 ↓
Testing
 ↓
Docker
```

The next stage of the project is to extend this foundation toward **production-oriented inference, multi-provider orchestration, stronger retrieval, memory, observability, and GPU serving**.