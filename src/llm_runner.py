import pandas as pd
import json
import time
from tqdm import tqdm
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from prompts import build_prompt

# -----------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------

MODEL_NAME = "gpt-4.1-mini"
INPUT_CSV = "data/merged.csv"
OUTPUT_CSV = "results/llm_results.csv"
START_INDEX = 0

LABELS = [
    "All-or-Nothing Thinking",
    "Emotional Reasoning",
    "Fortune-telling",
    "Labeling",
    "Magnification",
    "Mental filter",
    "Mind Reading",
    "No Distortion",
    "Overgeneralization",
    "Personalization",
    "Should statements",
]

# -----------------------------------------------------------------------
# 2. RETRY + OPENAI CALL
# -----------------------------------------------------------------------

@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=20))
def call_llm(prompt):
    client = OpenAI()

    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt
    )

    return response.output_text


# -----------------------------------------------------------------------
# 3. MAIN LOOP
# -----------------------------------------------------------------------

def run_llm():
    print("🔄 Loading dataset...")
    df = pd.read_csv(INPUT_CSV)

    results = []

    print("🚀 Starting LLM inference...")

    for i in tqdm(range(START_INDEX, len(df))):
        text = df.loc[i, "text"]

        prompt = build_prompt(text, LABELS)

        try:
            raw_output = call_llm(prompt)

            parsed = json.loads(raw_output)

            results.append({
                "index": i,
                "text": text,
                "extracted": parsed.get("extracted", ""),
                "reasoning": parsed.get("reasoning", ""),
                "predicted_label": parsed.get("label", "")
            })

        except Exception as e:
            print(f"⚠️ Error at row {i}: {e}")
            results.append({
                "index": i,
                "text": text,
                "extracted": "",
                "reasoning": "",
                "predicted_label": "ERROR"
            })

        if i % 50 == 0 and i > 0:
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
            print(f"💾 Progress saved at row {i}")

        time.sleep(0.3)

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print("🎉 All done! Saved to results/llm_results.csv")


# -----------------------------------------------------------------------
# 4. RUN
# -----------------------------------------------------------------------

if __name__ == "__main__":
    run_llm()
