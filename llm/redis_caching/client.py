# Asynchronous Example
import asyncio
from langcache import LangCache
import os 
from dotenv import load_dotenv 

load_dotenv()


langcache_client = LangCache(os.getenv("REDIS_DB_URI"),
                              cache_id = os.getenv('REDIS_CACHE_ID'), 
                              api_key = os.getenv('REDIS_KEY'))
