from llm.redis_caching.client import langcache_client
async def push_data(prompt : str, response: str, attributes : dict):

    if attributes:

        res = await langcache_client.set_async(prompt=prompt, 
                                            response=response, 
                                            attributes = attributes
                                            )
    else:
        res = await langcache_client.set_async(prompt=prompt, 
                                                    response=response, 
                    
                                                    )

    # Handle response
    print(res)
    return res 


async def flush_entries():
    langcache_client.flush()


async def delete_query_by_id(id: str):
    res = await langcache_client.delete_by_id_async(id = id)
    print(f"[REDIS LOG] DELETED ENTRY BY ID : {id}")


def search_query(input_prompt : str, threshold : float = None):
    if threshold is not None:
        res = langcache_client.search(prompt = input_prompt,
                                      similarity_threshold = threshold
                                      )
    else:
        res = langcache_client.search(prompt=input_prompt)


    return res

