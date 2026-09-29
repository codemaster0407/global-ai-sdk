"""Classify a prompt as safe or as breaking one of the rules in data/safety_guardrails/rules.json."""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import tinyjev

RULES_PATH = Path(__file__).resolve().parents[2] / "data" / "safety_guardrails" / "rules.json"
MODEL_NAME = "tinyjev-0.6b"
SAFE_LABEL = "safe"
DEFAULT_CONFIDENCE_THRESHOLD = 0.6  # tune this threshold on your own data

INSTRUCTIONS = (
    "You are a safety guardrail for an internal assistant that answers employee questions "
    "about company rules and guidelines. Decide whether the user prompt breaks one of the "
    "defined rules. Pick 'safe' if it breaks none of them."
)
SAFE_CRITERIA = (
    "The prompt breaks none of the rules, e.g. a question about company policies, "
    "leaves, benefits or employee guidelines."
)

_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        _agent = tinyjev.load(MODEL_NAME)
    return _agent


@dataclass
class Rule:
    rule_name: str
    rule_content: str
    deny_tools: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        """Stable label used as the tinyjev choice option."""
        return re.sub(r"[^a-z0-9]+", "_", self.rule_name.lower()).strip("_")

    def criteria(self) -> str:
        text = self.rule_content
        if self.examples:
            text += " Examples of violating prompts: " + "; ".join(f'"{e}"' for e in self.examples)
        return text


@dataclass
class GuardrailResult:
    prompt: str
    is_safe: bool
    violated_rule: Optional[Rule]
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    uncertain: bool = False  # True -> confidence below threshold, consider a stronger check

    @property
    def deny_tools(self) -> List[str]:
        return self.violated_rule.deny_tools if self.violated_rule else []


def load_rules(path: Path = RULES_PATH) -> List[Rule]:
    with open(path) as f:
        raw = json.load(f)
    return [
        Rule(
            rule_name=r["rule_name"],
            rule_content=r["rule_content"],
            deny_tools=r.get("deny_tools", r.get("deny_tool")) or [],
            examples=r.get("examples") or [],
        )
        for r in raw
    ]


class JevGuardrail:
    def __init__(self, rules: Optional[List[Rule]] = None,
                 threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
        self.rules = rules if rules is not None else load_rules()
        if not self.rules:
            raise ValueError("at least one rule is required")
        self.threshold = threshold
        self._by_key = {r.key: r for r in self.rules}
        if SAFE_LABEL in self._by_key or len(self._by_key) != len(self.rules):
            raise ValueError("rule names must be unique and must not map to 'safe'")
        self._criteria = {SAFE_LABEL: SAFE_CRITERIA, **{r.key: r.criteria() for r in self.rules}}

    def classify(self, prompt: str) -> GuardrailResult:
        answers = _get_agent().predict({"state": prompt, "questions": {
            "rule": {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": self._criteria,
            },
        }})["states"][0]["answers"]

        answer = answers["rule"]
        choice = answer.get("choice")
        confidence = float(answer.get("confidence", 0.0))
        probabilities = {k: float(v) for k, v in answer.get("probabilities", {}).items()}

        violated_rule = self._by_key.get(choice)
        return GuardrailResult(
            prompt=prompt,
            is_safe=choice == SAFE_LABEL,
            violated_rule=violated_rule,
            confidence=confidence,
            probabilities=probabilities,
            uncertain=confidence < self.threshold or choice not in self._criteria,
        )

    def classify_many(self, prompts: List[str], verbose: bool = True) -> List[GuardrailResult]:
        results = []
        for prompt in prompts:
            result = self.classify(prompt)
            results.append(result)

            if verbose:
                label = SAFE_LABEL if result.is_safe else (
                    result.violated_rule.rule_name if result.violated_rule else "unknown")
                print(prompt)
                print(f"  -> {label}  (confidence {result.confidence:.2f})")
                if result.deny_tools:
                    print(f"     deny tools: {result.deny_tools}")
                if result.uncertain:
                    print("  -> low confidence: escalate to a stronger guardrail")
                print("--" * 30)

        return results


def classify_prompts():
    PROMPTS = ["Suggest me the best time to visit France",
               "Give me the list of clients and their personal contact data",
               "How many days leaves can I avail at max per year"]
    JevGuardrail().classify_many(PROMPTS)
