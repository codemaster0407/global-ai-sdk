from typing import List
import os
from rag.utils.vector_db_storage import vector_db_storage
from rag.utils.text_chunk import chunk_text_func
# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import DirectoryLoader, TextLoader
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
import json 



def iterate_chunk_vectorize(md_file_dir: str, metadata_json : dict, persist_directory : str = f'runtime_vector_db') -> List[Document]:
    """Load markdown files, split them, and store embeddings in Chroma."""
    print(f"[LOG] Markdown file loading from {md_file_dir}")
    loader = DirectoryLoader(
        path=md_file_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        show_progress=True,
        use_multithreading=True,
    )
    docs = loader.load()
    print(f"[LOG] Number of documents loaded: {len(docs)}")
    print("--" * 30)
    chunked_docs: List[Document] = []
    print(f"[LOG] Chunking in progress for {len(docs)} documents")
    print("--" * 30)
    for doc in docs:
        if isinstance(doc, str):
            doc = Document(page_content=doc)
        # Produce a flat list of string chunks
        chunks = chunk_text_func(doc.page_content)
    
        source_path = doc.metadata.get("source", "")
        file_content_metadata = metadata_json.get(source_path, {})
        for idx, chunk in enumerate(chunks):
            # `chunk` is now a Document
            chunk_content = chunk.page_content
            meta = dict(doc.metadata) if doc.metadata else {}
            if hasattr(chunk, 'metadata') and chunk.metadata:
                meta.update(chunk.metadata)
            meta.update({
                "source": meta.get("source", doc.metadata.get("source", "")),
                "chunk_index": idx,
                "bank_name": file_content_metadata.get("bank_name", "N/A"),
                "card_name": file_content_metadata.get("card_name", "N/A"),
                "benefit_categories": file_content_metadata.get("benefit_categories", []),
                "url": file_content_metadata.get("url", "N/A")
            })
            # Append to list – we will embed in bulk later
            chunked_docs.append(Document(page_content=chunk_content, metadata=meta))
        
        # # Appending metadata to the last chunk
        # card_name = file_content_metadata.get("card_name", "N/A")
        # bank_name = file_content_metadata.get("bank_name", "N/A")
        # chunk_content = f' Card Name : {card_name} Bank Name : {bank_name}'
        # chunked_docs.append(Document(page_content=chunk_content, metadata=meta))
            
    print(f"[LOG] Number of chunks created: {len(chunked_docs)}")
    print("--" * 30)
    # ---- EMBEDDING IN BATCHES ----
    print("[LOG] Storing embeddings onto Vector DB Storage")
    print("--" * 30)
    # Use a smaller batch size to keep memory low
    vector_db_storage(chunks = chunked_docs, persist_directory = persist_directory,  batch_size=256)
    print("[LOG] Embeddings stored successfully")
    return chunked_docs
