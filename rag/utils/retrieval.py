from typing import List
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
# pyrefly: ignore [missing-import]
from langchain_chroma import Chroma
from rag.utils.embedding_helpers import nomic_embeddings

# ---------------------------------------------------------------------------
# Vector store utilities
# ---------------------------------------------------------------------------
def get_vector_store(persist_dir: str = "runtime_vector_db") -> Chroma:
    """Instantiate the Chroma vector store used by the pipeline.

    The store is created on‑the‑fly if the directory does not exist – this is
    safe because Chroma will initialise an empty collection.  ``persist_dir``
    matches the directory used in :func:`rag_helpers.vector_db_storage`.
    """
    return Chroma(persist_directory=persist_dir, embedding_function=nomic_embeddings)

# ---------------------------------------------------------------------------
# BM25 retriever utilities
# ---------------------------------------------------------------------------
# pyrefly: ignore [missing-import]
from langchain_community.retrievers import BM25Retriever

def get_bm25_retriever(docs: List[Document]) -> BM25Retriever:
    """Create a BM25 retriever from a list of ``Document`` objects.

    ``BM25Retriever.from_documents`` builds an internal inverted index that is
    fast to query and works entirely on CPU, making it a good complement to the
    dense vector store.
    """
    return BM25Retriever.from_documents(docs)

# ---------------------------------------------------------------------------
# Reranking utilities (cross‑encoder)
# ---------------------------------------------------------------------------
# pyrefly: ignore [missing-import]
from sentence_transformers import CrossEncoder

def get_cross_encoder(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
) -> CrossEncoder:
    """Load a cross‑encoder model for relevance reranking.

    The default model is lightweight (~100 MB) and provides good performance on
    short passages.  The ``CrossEncoder`` API expects a list of ``(query,
    passage)`` tuples and returns a relevance score for each pair.
    """
    return CrossEncoder(model_name)

# ---------------------------------------------------------------------------
# Hybrid retrieval entry point
# ---------------------------------------------------------------------------
def hybrid_retrieve(
    query: str,
    top_k: int = 5,
    vector_store: Chroma | None = None,
    bm25_retriever: BM25Retriever | None = None,
    rerank: bool = True,
    metadata_filter: dict | None = None,
) -> List[Document]:
    """Retrieve documents using a hybrid vector + BM25 approach.

    Parameters
    ----------
    query: str
        The user query.
    top_k: int, default 5
        Maximum number of documents to return after optional reranking.
    vector_store: Chroma | None
        If supplied, reuse an existing store; otherwise a new one is created.
    bm25_retriever: BM25Retriever | None
        Allows callers to pass a pre‑built BM25 retriever.  When ``None`` we
        load **all** documents from the vector store and build the BM25 index on the
        fly – this is cheap for the dataset size used in the assignment.
    rerank: bool, default True
        Whether to apply cross‑encoder reranking.  Setting ``False`` returns the
        raw union of the two retrieval methods.
    metadata_filter: dict | None
        Optional Chroma metadata filter, e.g. {"bank_name": "Chase"}.
        When provided, both vector search and BM25 results are filtered.

    Returns
    -------
    List[Document]
        A list of up to ``top_k`` ``Document`` objects sorted by relevance.
    """
    # 1️⃣ Vector similarity
    if vector_store is None:
        vector_store = get_vector_store()

    search_kwargs = {"k": top_k}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    vector_results = vector_store.similarity_search(query, **search_kwargs)

    # 2️⃣ BM25 lexical search
    if bm25_retriever is None:
        # vector_store.get() returns a dict with keys: ids, documents, metadatas, …
        # We must convert it into a list of Document objects for BM25Retriever.
        get_kwargs = {}
        if metadata_filter:
            get_kwargs["where"] = metadata_filter
        raw = vector_store.get(**get_kwargs)
        all_docs = [
            Document(page_content=txt, metadata=meta or {})
            for txt, meta in zip(raw["documents"], raw["metadatas"])
        ]
        if not all_docs:
            # No documents match the filter — return vector results only
            return vector_results[:top_k], [0.0] * min(top_k, len(vector_results))
        bm25_retriever = get_bm25_retriever(all_docs)
    bm25_results = bm25_retriever.invoke(query)[:top_k]
    # print(len(bm25_results))

    # 3️⃣ Combine & deduplicate (preserve vector order first)
    combined: dict[str, Document] = {}
    for doc in vector_results:
        key = f"{doc.metadata.get('source','')}::{doc.metadata.get('chunk_index','')}"
        combined[key] = doc
    for doc in bm25_results:
        key = f"{doc.metadata.get('source','')}::{doc.metadata.get('chunk_index','')}"
        combined.setdefault(key, doc)  # keep the vector version if duplicate

    candidates = list(combined.values())

    if not rerank:
        docs = candidates[:top_k]
        return docs, [0.0] * len(docs)

    # 4️⃣ Rerank with cross‑encoder
    cross_encoder = get_cross_encoder()
    pairs = [(query, doc.page_content) for doc in candidates]
    scores = cross_encoder.predict(pairs)

    # Sort by descending score
    ranked = [
        doc for _, doc in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    ]
    return ranked[:top_k], sorted(scores, reverse = True)[: top_k ]


def baseline_retrieval_function(query: str, top_k : int):
    vector_store = get_vector_store()
    vector_results = vector_store.similarity_search(query, k=top_k)
    print(vector_results)
    return vector_results
# ---------------------------------------------------------------------------
# Convenience wrapper used by downstream notebooks / scripts
# ---------------------------------------------------------------------------
def advanced_retrieve_function(query: str, top_k: int , rerank: bool = False, metadata_filter: dict | None = None) -> List[Document]:
    """Simple wrapper that hides the optional arguments.

    Example
    -------
    ```python
    from rag_helpers.retrieval import retrieve
    docs = retrieve("What are the main challenges of RAG?", top_k=3)
    for d in docs:
        print(d.metadata["source"], "-", d.page_content[:200])
    ```
    """
    baseline_retrieval_docs = baseline_retrieval_function(query, top_k)
    hybrid_retrieval_docs, hybrid_scores =  hybrid_retrieve(query, top_k=top_k, rerank=rerank, metadata_filter=metadata_filter)


    return baseline_retrieval_docs + hybrid_retrieval_docs


