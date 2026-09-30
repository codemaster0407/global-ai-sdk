from functools import lru_cache
import os

from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()


@lru_cache(maxsize=None)
def initiate_pinecone_client(index_name='ai-sdk-testing'):
    '''
        Handle to a Pinecone index. Created on first use and reused, one per index name.
    '''
    api_key = os.getenv('PINECONE_KEY')
    if not api_key:
        raise RuntimeError('PINECONE_KEY is not set in the environment / .env')
    pc = Pinecone(api_key=api_key)
    return pc.Index(index_name)
