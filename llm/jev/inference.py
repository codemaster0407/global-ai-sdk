"""Route a prompt to the right tool before calling a bigger model."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import tinyjev

agent = tinyjev.load("tinyjev-0.6b")

DEFAULT_CONFIDENCE_THRESHOLD = 0.6  # tune this threshold on your own data


@dataclass
class ToolChoice:
    prompt: str
    choice: Optional[str]
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    fallback: bool = False  # True -> let the LLM do its own tool calling

    def ranked(self, top_k: Optional[int] = None) -> List[tuple]:
        """Options sorted by probability, highest first."""
        items = sorted(self.probabilities.items(), key=lambda kv: kv[1], reverse=True)
        return items[:top_k] if top_k else items


def _validate(tools: List[str], criteria: dict) -> None:
    if not tools:
        raise ValueError("tools must be a non-empty list")
    missing = [t for t in tools if t not in criteria]
    if missing:
        raise ValueError(f"criteria is missing entries for tools: {missing}")


def classify_prompt(
    prompt: str,
    tools: List[str],
    instructions: str,
    criteria: dict,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> ToolChoice:
    """Pick one tool for a single prompt using a tinyjev choice question."""
    answers = agent.predict({"state": prompt, "questions": {
        "tool": {
            "type": "choice",
            "instructions": instructions,
            "criteria": {t: criteria[t] for t in tools},
        },
    }})["states"][0]["answers"]

    tool = answers["tool"]
    choice = tool.get("choice")
    confidence = float(tool.get("confidence", 0.0))
    probabilities = {k: float(v) for k, v in tool.get("probabilities", {}).items()}

    # Fall back when the model is unsure or picks something outside the allowed tools.
    fallback = confidence < threshold or choice not in tools
    return ToolChoice(
        prompt=prompt,
        choice=None if choice not in tools else choice,
        confidence=confidence,
        probabilities=probabilities,
        fallback=fallback,
    )


def classify_prompts(
    prompts: List[str],
    tools: List[str],
    instructions: str,
    criteria: dict,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    verbose: bool = True,
) -> List[ToolChoice]:
    """Classify each prompt into one of `tools`; returns one ToolChoice per prompt."""
    _validate(tools, criteria)

    results = []
    for prompt in prompts:
        result = classify_prompt(prompt, tools, instructions, criteria, threshold)
        results.append(result)

        if verbose:
            print(prompt)
            print(f"  -> {result.choice}  (confidence {result.confidence:.2f})")
            for option, p in result.ranked():
                print(f"     {option:<9} {p:.2f}")
            if result.fallback:
                print("  -> low confidence: fall back to the LLM's own tool calling")
            print("--" * 30)

    return results
