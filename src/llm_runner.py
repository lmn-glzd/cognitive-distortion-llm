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

MODEL_NAME = "gpt-4o-mini"   # istəsən sonra dəyişərsən
INPUT_CSV = "data/merged.csv"
OUTPUT_CSV = "results/llm_results.csv"
START_INDEX = 0   # qaldığın yerdən davam etmək üçün dəyişə bilərsən

LABELS = [
    "Overgeneralization",
    "Catastrophizing",
    "Personalization",
    "Mind Reading",
    "Emotional Reasoning",
    "Should Statements",
    "Labeling",
    "Black-and-White Thinking",
]
# Burada öz datasetində olan distortions yazılmalıdır.


# -----------------------------------------------------------------------
# 2. OPENAI CLIENT
# -----------------------------------------------------------------------

client = OpenAI()


# -----------------------------------------------------------------------
# 3. RETRY MECHANISM FOR OPENAI CALLS
# -----------------------------------------------------------------------

@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=20))
def call_llm(prompt):
    """
    Sends prompt to OpenAI with automatic retries.
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message["content"]


# -----------------------------------------------------------------------
# 4. MAIN LLM PROCESSING LOOP
# -----------------------------------------------------------------------

def run_llm():
    print("🔄 Loading dataset...")
    df = pd.read_csv(INPUT_CSV)

    results = []

    print("🚀 Starting LLM inference...")

    for i in tqdm(range(START_INDEX, len(df))):
        text = df.loc[i, "text"]

        # Build ERD-style prompt
        prompt = build_prompt(text, LABELS)

        try:
            raw_output = call_llm(prompt)

            # try to parse JSON output
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

        # save progress every 50 samples
        if i % 50 == 0 and i > 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(OUTPUT_CSV, index=False)
            print(f"💾 Progress saved at row {i}")

        time.sleep(0.3)  # small delay to avoid rate limits

    # Final save
    final_df = pd.DataFrame(results)
    final_df.to_csv(OUTPUT_CSV, index=False)
    print("🎉 All done! Saved to results/llm_results.csv")


# -----------------------------------------------------------------------
# 5. RUN
# -----------------------------------------------------------------------

if __name__ == "__main__":
    run_llm()
