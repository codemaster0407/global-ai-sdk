"""Route a prompt to the right tool before calling a bigger model."""
import tinyjev

agent = tinyjev.load("tinyjev-0.6b")

PROMPTS = [
    "Give me the gold price in India today.",
    "Who is the manager of the data science team?",
    "Explain what a vector database is.",
]

for prompt in PROMPTS:
    answers = agent.predict({"state": prompt, "questions": {
        "tool": {
            "type": "choice",
            "instructions": "Which tool is needed to answer this request?",
            "criteria": {
                "internet": "Needs current or live information from the web (prices, news, weather)",
                "database": "Needs information about people in the organisation",
                "none":     "Can be answered from general knowledge without any tool",
            },
        },
    }})["states"][0]["answers"]

    tool = answers["tool"]
    print(prompt)
    print(f"  -> {tool['choice']}  (confidence {tool['confidence']})")
    for option, p in tool["probabilities"].items():
        print(f"     {option:<9} {p:.2f}")

    if tool["confidence"] < 0.6:        # tune this threshold on your own data
        print("  -> low confidence: fall back to the LLM's own tool calling")
    print("--" * 30)