'''
Upsert LangChain ``Document`` chunks into a Pinecone index with integrated embedding.

The index embeds the text server-side (``llama-text-embed-v2``), so each record carries
the raw text in the field named by the index's ``field_map`` plus flat metadata; no
vectors are computed here.
'''

import hashlib
import re
import time
from typing import Any, Iterable, List

from langchain_core.documents import Document
from pinecone import ApiError

from vector_db.pinecone_func.client import initiate_pinecone_client

# Must match the index's embed.field_map["text"] ("page_content" for ai-sdk-testing)
TEXT_FIELD = 'page_content'
# Pinecone's cap on records per upsert_records call when it embeds them itself
MAX_RECORDS_PER_BATCH = 96
DEFAULT_NAMESPACE = '__default__'
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _record_id(metadata: dict, text: str) -> str:
    '''
        Stable id so re-running the upsert overwrites chunks instead of duplicating them:
        "<source>#<chunk_index>" when both are known, else a hash of the text.
    '''
    source, chunk_index = metadata.get('source'), metadata.get('chunk_index')
    if source and chunk_index is not None:
        raw = f'{source}#{chunk_index}'
    else:
        raw = hashlib.sha256(text.encode('utf-8')).hexdigest()[:32]
    # Keep ids ASCII and short (Pinecone allows up to 512 chars)
    return re.sub(r'[^A-Za-z0-9._#/-]', '_', raw)[-512:]


def _clean_metadata(metadata: dict) -> dict:
    '''
        Pinecone metadata must be flat: str, int, float, bool or a list of strings. Null is not allowed.
    '''
    cleaned = {}
    for key, value in metadata.items():
        if value is None or key in (TEXT_FIELD, '_id', 'id'):
            continue
        if isinstance(value, (str, bool, int, float)):
            cleaned[key] = value
        elif isinstance(value, (list, tuple, set)):
            cleaned[key] = [str(v) for v in value if v is not None]
        else:
            cleaned[key] = str(value)
    return cleaned


def chunks_to_records(chunked_docs: Iterable[Document]) -> List[dict[str, Any]]:
    '''
        Convert Documents into Pinecone records: {"_id", "page_content", **metadata}.
    '''
    records, seen_ids = [], set()
    for doc in chunked_docs:
        text = doc.page_content.strip()
        if not text:
            continue
        record_id = _record_id(doc.metadata or {}, text)
        if record_id in seen_ids:
            raise ValueError(f'Duplicate record id {record_id!r}: two chunks share source and chunk_index')
        seen_ids.add(record_id)
        records.append({'_id': record_id, TEXT_FIELD: text, **_clean_metadata(doc.metadata or {})})
    return records


def _upsert_with_retry(pc_index, batch: List[dict], namespace: str, max_retries: int) -> None:
    for attempt in range(max_retries + 1):
        try:
            pc_index.upsert_records(namespace=namespace, records=batch)
            return
        except ApiError as e:
            if e.status_code not in RETRYABLE_STATUS or attempt == max_retries:
                raise
            delay = min(2 ** attempt, 60)
            print(f'[PINECONE] {e.status_code} on upsert, retrying in {delay}s ({attempt + 1}/{max_retries})')
            time.sleep(delay)


def insert_chunks_pinecone(
    chunked_docs: Iterable[Document],
    namespace: str = DEFAULT_NAMESPACE,
    index_name: str = 'ai-sdk-testing',
    batch_size: int = MAX_RECORDS_PER_BATCH,
    max_retries: int = 5,
) -> int:
    '''
        Upsert chunks in batches of up to 96, retrying rate limits and server errors.
        Returns the number of records sent. Upserts are eventually consistent: new records
        can take a few seconds to show up in search results and index stats.
    '''
    if not 1 <= batch_size <= MAX_RECORDS_PER_BATCH:
        raise ValueError(f'batch_size must be between 1 and {MAX_RECORDS_PER_BATCH}')

    records = chunks_to_records(chunked_docs)
    if not records:
        print('[PINECONE] No non-empty chunks to upsert')
        return 0

    pc_index = initiate_pinecone_client(index_name=index_name)
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        _upsert_with_retry(pc_index, batch, namespace, max_retries)
        print(f'[PINECONE] Upserted {start + len(batch)}/{len(records)} records into {index_name!r}/{namespace!r}')
    return len(records)
