FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-runtime.txt .

RUN pip install --no-cache-dir -r requirements-runtime.txt

COPY gateway ./gateway
COPY inference ./inference
COPY retrieval ./retrieval

COPY experiments/faiss.index ./experiments/faiss.index
COPY experiments/faiss_metadata.json ./experiments/faiss_metadata.json

EXPOSE 8000

CMD ["uvicorn", "inference.server:app", "--host", "0.0.0.0", "--port", "8000"]