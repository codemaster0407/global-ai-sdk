from typing import List, Literal, Optional

from dotenv import load_dotenv
from langfuse import get_client
import os 

load_dotenv()

# Reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_HOST from the environment
langfuse = get_client()


def create_prompt(
    name: str,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    prompt_type: Literal["chat", "text"] = "chat",
    labels: Optional[List[str]] = None,
):
    """Create a prompt in Langfuse, or a new version if ``name`` already exists.

    ``chat`` prompts are stored as a message list (system first, then user);
    ``text`` prompts are stored as a single string.  Use ``{{variable}}``
    placeholders in the prompt text and fill them later with ``compile()``.
    """
    if prompt_type == "chat":
        prompt = []
        if system_prompt:
            prompt.append({"role": "system", "content": system_prompt})
        prompt.append({"role": "user", "content": user_prompt})
    elif prompt_type == "text":
        prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
    else:
        raise ValueError(f"prompt_type must be 'chat' or 'text', got '{prompt_type}'")

    return langfuse.create_prompt(
        name=name,
        type=prompt_type,
        prompt=prompt,
        labels=labels if labels is not None else ["production"],
    )


def search_prompt(name : str):
    try:
        return langfuse.get_prompt(name = name)

    except Exception as e:
        print(f'[Langfuse Client ERROR] Error at {os.getcwd()} with {e}')