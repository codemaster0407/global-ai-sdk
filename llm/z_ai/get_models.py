from llm.z_ai.client import zai_client 


def list_models():
    response = zai_client.get(f'/models', cast_type = object)
    return [model['id'] for model in response['data']]