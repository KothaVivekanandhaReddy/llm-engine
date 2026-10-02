from retrieval.retrieve import Retriever


def test_kidney_query_returns_kidney_function():
    retriever = Retriever()

    results = retriever.search(
        "What does the kidney do?",
        top_k=3,
    )

    assert len(results) == 3
    assert results[0]["title"] == "Kidney Function"


def test_retrieval_returns_scores():
    retriever = Retriever()

    results = retriever.search(
        "What does the kidney do?",
        top_k=3,
    )

    assert all("score" in result for result in results)
    assert all(isinstance(result["score"], float) for result in results)


def test_retrieval_respects_top_k():
    retriever = Retriever()

    results = retriever.search(
        "What does the kidney do?",
        top_k=2,
    )

    assert len(results) == 2