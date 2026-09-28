from typing import Dict, Optional

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from rag.eval.zai_llm import glm_model

judge = glm_model()


def evaluate_generated_output(
    generated_output: str,
    original_content: str,
    query: Optional[str] = None,
    threshold: float = 0.7,
) -> Dict[str, dict]:
    """Score a RAG answer against the source content it was generated from.

    - Faithfulness: are the claims in ``generated_output`` supported by ``original_content``?
    - Answer relevancy (only when ``query`` is given): does the output answer the question?
    """
    test_case = LLMTestCase(
        input=query or "",
        actual_output=generated_output,
        retrieval_context=[original_content],
    )

    metrics = [FaithfulnessMetric(threshold=threshold, model=judge, include_reason=True)]
    if query:
        metrics.append(AnswerRelevancyMetric(threshold=threshold, model=judge, include_reason=True))

    results = {}
    for metric in metrics:
        metric.measure(test_case)
        results[metric.__name__] = {
            "score": metric.score,
            "passed": metric.is_successful(),
            "reason": metric.reason,
        }
        print(f"{metric.__name__}: {metric.score} -> {metric.reason}")

    return results
