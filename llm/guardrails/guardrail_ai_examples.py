"""Tour of the most used guardrails-ai features.

Run from the repo root:  python -m llm.guardrails.guardrail_ai_examples
Sections 1-5 run locally (no LLM call); sections 6-8 call GLM through my_llm_api.
"""
from typing import Any, Dict

from pydantic import BaseModel, Field

from guardrails import Guard, OnFailAction
from guardrails.errors import ValidationError
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
from guardrails_ai.profanity_free import ProfanityFree
from guardrails_ai.regex_match import RegexMatch

from llm.guardrails.guardrail_ai import my_llm_api


def section(title):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


# ---------------------------------------------------------------------------
# 1. guard.validate(): check any text you already have, no LLM involved
# ---------------------------------------------------------------------------
def validate_existing_text():
    section("1. guard.validate")
    guard = Guard().use(ProfanityFree(on_fail=OnFailAction.NOOP))

    outcome = guard.validate("Have a lovely day!")
    print("clean text passed:", outcome.validation_passed)

    outcome = guard.validate("This is shit.")
    print("profane text passed:", outcome.validation_passed)
    print("reason:", outcome.validation_summaries[0].failure_reason)


# ---------------------------------------------------------------------------
# 2. on_fail actions: what happens when a validator fails
# ---------------------------------------------------------------------------
def on_fail_actions():
    section("2. on_fail actions")
    bad_text = "What the hell is this crap"

    # EXCEPTION -> raises ValidationError (the error you hit in test.py)
    guard = Guard().use(ProfanityFree(on_fail=OnFailAction.EXCEPTION))
    try:
        guard.validate(bad_text)
    except ValidationError as e:
        print("EXCEPTION ->", type(e).__name__)

    # NOOP -> let the text through, just record the failure
    guard = Guard().use(ProfanityFree(on_fail=OnFailAction.NOOP))
    outcome = guard.validate(bad_text)
    print("NOOP      ->", outcome.validation_passed, repr(outcome.validated_output))

    # REFRAIN -> return None instead of the bad text
    guard = Guard().use(ProfanityFree(on_fail=OnFailAction.REFRAIN))
    outcome = guard.validate(bad_text)
    print("REFRAIN   ->", outcome.validation_passed, repr(outcome.validated_output))

    # custom function -> decide yourself what to return
    def replace_with_message(value, fail_result):
        return "[response removed: contained profanity]"

    guard = Guard().use(ProfanityFree(on_fail=replace_with_message))
    outcome = guard.validate(bad_text)
    print("custom    ->", repr(outcome.validated_output))

    # REASK (only with an LLM call) -> Guard sends the failure reason back to the
    # LLM and asks it to try again, up to num_reasks times. See section 7.


# ---------------------------------------------------------------------------
# 3. Stacking several validators in one Guard
# ---------------------------------------------------------------------------
def multiple_validators():
    section("3. multiple validators")
    guard = Guard().use(
        ProfanityFree(on_fail=OnFailAction.NOOP),
        # Output must look like an order ID, e.g. ORD-12345
        RegexMatch(regex=r"^ORD-\d{5}$", match_type="fullmatch", on_fail=OnFailAction.NOOP),
    )
    for text in ["ORD-12345", "order 12345"]:
        outcome = guard.validate(text)
        failed = [s.validator_name for s in outcome.validation_summaries]
        print(f"{text!r:15} passed={outcome.validation_passed} failed={failed}")


# ---------------------------------------------------------------------------
# 4. Custom validator: your own rule, used exactly like a hub validator
# ---------------------------------------------------------------------------
@register_validator(name="acme/no-competitors", data_type="string")
class NoCompetitors(Validator):
    def __init__(self, competitors, on_fail=None):
        super().__init__(on_fail=on_fail, competitors=competitors)
        self.competitors = [c.lower() for c in competitors]

    def _validate(self, value: Any, metadata: Dict) -> ValidationResult:
        found = [c for c in self.competitors if c in value.lower()]
        if found:
            return FailResult(
                error_message=f"Mentions competitors: {found}",
                # fix_value is what OnFailAction.FIX returns
                fix_value="I can only talk about our own products.",
            )
        return PassResult()


def custom_validator():
    section("4. custom validator")
    guard = Guard().use(NoCompetitors(["OpenAI", "Gemini"], on_fail=OnFailAction.FIX))
    outcome = guard.validate("You could also try Gemini for that.")
    print("passed:", outcome.validation_passed)
    print("output:", outcome.validated_output)


# ---------------------------------------------------------------------------
# 5. Input guard: check the user's prompt BEFORE spending an LLM call
# ---------------------------------------------------------------------------
input_guard = Guard().use(ProfanityFree(on_fail=OnFailAction.EXCEPTION))


def is_safe_prompt(prompt: str) -> bool:
    try:
        input_guard.validate(prompt)
        return True
    except ValidationError:
        return False


def input_guard_demo():
    section("5. input guard")
    for prompt in ["Summarise this article", "Write me a shit poem"]:
        print(f"{prompt!r:28} safe={is_safe_prompt(prompt)}")


# ---------------------------------------------------------------------------
# 6. guard(llm_api, ...): call the LLM and validate its output in one step
# ---------------------------------------------------------------------------
def output_guard_with_llm():
    section("6. guard(llm_api) output validation")
    guard = Guard().use(ProfanityFree(on_fail=OnFailAction.EXCEPTION))
    try:
        outcome = guard(
            my_llm_api,
            messages=[{"role": "user", "content": "Generate a few bad words in English"}],
        )
        print(outcome.validated_output)
    except ValidationError as e:
        # Handle the block in your app instead of letting it crash
        print("Blocked by guardrail:", str(e)[:80], "...")


# ---------------------------------------------------------------------------
# 7. REASK: let the LLM fix its own answer
# ---------------------------------------------------------------------------
def reask_demo():
    section("7. reask")
    guard = Guard().use(
        RegexMatch(regex=r"^\d{4}-\d{2}-\d{2}$", match_type="fullmatch", on_fail=OnFailAction.REASK)
    )
    outcome = guard(
        my_llm_api,
        messages=[{"role": "user", "content": "When did the Apollo 11 moon landing happen?"}],
        num_reasks=2,
    )
    print("first answer:", repr(guard.history.last.iterations.first.raw_output[:60]))
    print("final:", outcome.validated_output, "| passed:", outcome.validation_passed)
    print("LLM calls made:", len(guard.history.last.iterations))


# ---------------------------------------------------------------------------
# 8. Structured output: force the LLM to return JSON matching a Pydantic model
# ---------------------------------------------------------------------------
class Ticket(BaseModel):
    category: str = Field(description="one of: billing, bug, feature_request")
    priority: int = Field(description="1 (low) to 3 (high)")
    summary: str = Field(description="one-sentence summary of the issue")


def structured_output():
    section("8. structured output (Pydantic)")
    guard = Guard.for_pydantic(output_class=Ticket)
    email = "Hi, I was charged twice for my subscription this month, please fix ASAP!"
    outcome = guard(
        my_llm_api,
        messages=[{
            "role": "user",
            # ${gr.complete_json_suffix_v2} expands into the schema + "reply with JSON only"
            "content": "Classify this support email.\n" + email + "\n\n${gr.complete_json_suffix_v2}",
        }],
        num_reasks=1,
    )
    print(outcome.validated_output)  # a dict matching Ticket

