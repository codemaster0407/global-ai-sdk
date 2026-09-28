# To run this code you need to install the following dependencies:
# pip install google-genai

import os
import re
import time
from google import genai
from google.genai import errors, types
from dotenv import load_dotenv

load_dotenv()

MAX_RETRIES = 5
RETRYABLE_CODES = {429, 500, 503, 504}


def _retry_delay(error, attempt):
    '''
        Seconds to wait before retrying: the delay suggested in the error, else exponential backoff.
    '''
    # 429 errors carry a google.rpc.RetryInfo detail, e.g. {"retryDelay": "23s"}
    body = error.details if isinstance(error.details, dict) else {}
    for detail in body.get('error', {}).get('details', []):
        if isinstance(detail, dict) and detail.get('@type', '').endswith('RetryInfo') and 'retryDelay' in detail:
            # retryDelay is rounded down to whole seconds, so wait 1s extra
            return float(detail['retryDelay'].rstrip('s')) + 1

    # Fallback: "Please retry in 23.45s." in the error message
    match = re.search(r'retry in ([\d.]+)\s*s', error.message or '', re.IGNORECASE)
    if match:
        return float(match.group(1)) + 1

    return min(2 ** attempt, 60)


def generate(user_prompt, system_prompt):
    try:
        client = genai.Client(
            api_key=os.environ.get("GOOGLE_AI_STUDIO_KEY"),
        )

        model = "gemini-flash-lite-latest"
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=user_prompt),
                ],
            ),
        ]
        tools = [
            types.Tool(googleSearch=types.GoogleSearch(
            )),
        ]
        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level="MINIMAL",
            ),
            audio_transcription_config=types.AudioTranscriptionConfig(
            ),
            tools=tools,
        )

        for attempt in range(MAX_RETRIES + 1):
            try:
                for chunk in client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
                ):
                    if text := chunk.text:
                        print(text, end="")
                return
            except errors.APIError as e:
                if e.code not in RETRYABLE_CODES or attempt == MAX_RETRIES:
                    raise
                delay = _retry_delay(e, attempt)
                print(f"[GEMINI LOG] {e.code} {e.status}: retrying in {delay:.1f}s (retry {attempt + 1}/{MAX_RETRIES})")
                time.sleep(delay)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    generate()
