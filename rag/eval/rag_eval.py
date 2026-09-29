from typing import Dict, Optional

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
import json 
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




def precision_at_k(retrieved_sources: list[str], relevant_sources: list[str], k: int) -> float:
    """
    Precision@k = (unique relevant docs in top-k) / k
    """
    if k == 0:
        return 0.0
        
    top_k = retrieved_sources[:k]
    relevant_set = set(relevant_sources)
    
    # Use a set intersection to count unique relevant documents found
    unique_hits = len(set(top_k).intersection(relevant_set))
    
    return unique_hits / k


def recall_at_k(retrieved_sources: list[str], relevant_sources: list[str], k: int) -> float:
    """
    Recall@k = (unique relevant docs in top-k) / (total relevant docs)
    """
    if not relevant_sources:
        return 1.0  # If there are no relevant docs expected, recall is trivially 100%
        
    top_k = retrieved_sources[:k]
    relevant_set = set(relevant_sources)
    # Use a set intersection to prevent recall from going over 1.0 if there are duplicates
    unique_hits = len(set(top_k).intersection(relevant_set))
    
    return unique_hits / len(relevant_set)




def hit_rate_at_k(retrieved_sources: list[str], relevant_sources: list[str], k: int) -> float:
    """
    Hit Rate@k = 1.0 if at least one relevant doc is in top-k, else 0.0
    """
    if k == 0 or not relevant_sources:
        return 0.0
        
    top_k = retrieved_sources[:k]
    relevant_set = set(relevant_sources)
    
    for source in top_k:
        if source in relevant_set:
            return 1.0
            
    return 0.0


def mrr(retrieved_sources: list[str], relevant_sources: list[str]) -> float:
    """
    Mean Reciprocal Rank (MRR): 1 / rank of the first relevant document
    """
    if not relevant_sources:
        return 0.0
        
    relevant_set = set(relevant_sources)
    for idx, source in enumerate(retrieved_sources):
        if source in relevant_set:
            return 1.0 / (idx + 1)
            
    return 0.0


