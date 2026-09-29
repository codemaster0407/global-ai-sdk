import os
from datetime import timedelta

import pandas as pd
from dotenv import load_dotenv
from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
from evidently.presets import TextEvals
from evidently.tests import lte, gte, eq
from evidently.descriptors import LLMEval, TestSummary, DeclineLLMEval, Sentiment, TextLength, IncludesWords
from evidently.llm.templates import BinaryClassificationPromptTemplate
from evidently.llm.utils.wrapper import OpenAIOptions, RateLimits

load_dotenv()

# Evidently's LLM judges speak the OpenAI API, and Z.ai exposes an
# OpenAI-compatible endpoint, so GLM can be used as the judge model.
JUDGE_PROVIDER = "openai"
JUDGE_MODEL = "glm-4.5-flash"
JUDGE_OPTIONS = OpenAIOptions(
    api_key=os.getenv("Z_AI_API_KEY"),
    api_url="https://api.z.ai/api/paas/v4/",
    # Evidently fires all rows in parallel by default; Z.ai answers that with
    # 429 "Rate limit reached". This sends one judge request at a time.
    limits=RateLimits(rpm=1, interval=timedelta(seconds=1)),
)

data = [
    ["What is the chemical symbol for gold?", "Gold chemical symbol is Au."],
    ["What is the capital of Japan?", "The capital of Japan is Tokyo."],
    ["Tell me a joke.", "Why don't programmers like nature? Too many bugs!"],
    ["When does water boil?", "Water's boiling point is 100 degrees Celsius."],
    ["Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa."],
    ["What’s the fastest animal on land?", "The cheetah is the fastest land animal, capable of running up to 75 miles per hour."],
    ["Can you help me with my math homework?", "I'm sorry, but I can't assist with homework."],
    ["How many states are there in the USA?", "USA has 50 states."],
    ["What’s the primary function of the heart?", "The primary function of the heart is to pump blood throughout the body."],
    ["Can you tell me the latest stock market trends?", "I'm sorry, but I can't provide real-time stock market trends. You might want to check a financial news website or consult a financial advisor."]
]
columns = ["question", "answer"]

eval_df = pd.DataFrame(data, columns=columns)
#eval_df.head()


# Custom LLM judge: describe the criteria in plain language, get a label per row
conciseness = BinaryClassificationPromptTemplate(
    pre_messages=[("system", "You are a judge evaluating the responses of a chatbot.")],
    criteria="""A CONCISE answer is short and to the point: it answers the question directly
    without unnecessary detail or filler.
    A VERBOSE answer adds details, explanations or suggestions the user did not ask for.""",
    target_category="CONCISE",
    non_target_category="VERBOSE",
    uncertainty="unknown",
    include_reasoning=True,
)


def build_dataset(df: pd.DataFrame) -> Dataset:
    """Wrap a question/answer DataFrame and score every row with descriptors.

    Descriptors add one new column per check. Rule-based ones run locally;
    the *LLMEval ones call the judge model once per row.
    """
    return Dataset.from_pandas(
        df,
        data_definition=DataDefinition(text_columns=["question", "answer"]),
        descriptors=[
            # Local, deterministic checks
            TextLength("answer", alias="Length", tests=[lte(150)]),
            Sentiment("answer", alias="Sentiment", tests=[gte(0)]),
            IncludesWords("answer", words_list=["sorry", "apologize"], alias="Denials",
                          tests=[eq(False)]),
            # LLM-as-a-judge: built-in refusal check
            DeclineLLMEval("answer", provider=JUDGE_PROVIDER, model=JUDGE_MODEL,
                           alias="Refusal", tests=[eq("OK", column="Refusal")]),
            # LLM-as-a-judge: our own criteria from the template above
            LLMEval("answer", template=conciseness, provider=JUDGE_PROVIDER, model=JUDGE_MODEL,
                    alias="Conciseness", tests=[eq("CONCISE", column="Conciseness")]),
            # One pass/fail column per row that combines all tests above
            TestSummary(success_all=True, alias="All tests passed"),
        ],
        options=JUDGE_OPTIONS,
    )


def run_report(dataset: Dataset, html_path: str = "evaluation/llm_observability_report.html"):
    """Aggregate row scores into a report (distributions + test pass rates)."""
    report = Report([TextEvals()], include_tests=True)
    snapshot = report.run(dataset, None)
    snapshot.save_html(html_path)
    return snapshot

def run_observability():
    dataset = build_dataset(eval_df)

    # Row-level results: the original columns plus one column per descriptor
    scored = dataset.as_dataframe()
    pd.set_option("display.max_columns", None, "display.width", 200)
    print(scored.drop(columns=["question"]))

    snapshot = run_report(dataset)
    print("\nReport saved to evaluation/llm_observability_report.html")
