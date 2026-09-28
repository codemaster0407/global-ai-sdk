import asyncio
import json
from typing import Optional, Type

from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel

from llm.z_ai.client import zai_client


class glm_model(DeepEvalBaseLLM):
    """GLM (Z.ai) as the judge model for DeepEval metrics."""

    def __init__(self, model_name: str = 'glm-5.3'):
        self.model_name = model_name
        super().__init__(model_name)

    def load_model(self):
        return zai_client

    def generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None):
        # DeepEval passes a pydantic `schema` when it expects structured output
        # (verdicts, statements, reasons); return a parsed instance in that case.
        extra = {}
        if schema is not None:
            prompt = (
                f'{prompt}\n\nRespond ONLY with a JSON object that matches this JSON schema:\n'
                f'{json.dumps(schema.model_json_schema())}'
            )
            extra["response_format"] = {"type": "json_object"}

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            thinking={
                "type": "enabled",  # Optional: "enabled" only, reasoning is always enabled
                "clearThinking": True
            },
            max_tokens=4096,
            temperature=0.1,
            **extra,
        )
        content = response.choices[0].message.content

        if schema is not None:
            return schema.model_validate_json(content)
        return content

    async def a_generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None):
        # The Z.ai client is synchronous; run it in a thread so metrics can evaluate concurrently
        return await asyncio.to_thread(self.generate, prompt, schema)

    def get_model_name(self):
        return self.model_name
