from typing import Optional

from llm.z_ai.client import zai_client
from guardrails import Guard, OnFailAction
from guardrails.errors import ValidationError
from guardrails_ai.profanity_free import ProfanityFree

guard = Guard().use(ProfanityFree(on_fail=OnFailAction.EXCEPTION))

def my_llm_api(*, messages: Optional[list[dict]] = None, **kwargs) -> str:
    """Custom LLM API wrapper.

    At least one of messages should be provided.

    Args:
        messages: Chat messages to send to the LLM
        **kwargs: Any additional arguments to be passed to the LLM API

    Returns:
        str: The output of the LLM API
    """
    if messages == []:
        raise Exception(f'Error : No messages for the LLM to generate results on....')
    
    llm_output = zai_client.chat.completions.create(
            model='glm-5.3',
            messages=messages,
            thinking={
                "type": "enabled",  # Optional: "enabled" only, reasoning is always enabled
                "clearThinking" : True
            },
            reasoning_effort="max",
            max_tokens=4096,
            temperature=0.1,
        )

    # Guard validates a string, not the SDK response object
    return llm_output.choices[0].message.content

# Wrap your LLM API call

def profanity_check(message, **kwargs):

    try:
        validated_response = guard(
            my_llm_api,
            messages=[{"role":"user","content":message}], **kwargs
        )
    except ValidationError as e:
        # The LLM's answer contained profanity, so the guard blocked it
        print(f'Blocked by guardrail: {e}')
        return None
    print(validated_response.validated_output)
    return validated_response.validated_output
