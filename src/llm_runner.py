import pandas as pd
import json
import time
from pathlib import Path

from tqdm import tqdm
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

from prompts import build_prompt

# -----------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------

MODEL_NAME = "gpt-4o-mini"
INPUT_CSV = "data/merged.csv"
OUTPUT_CSV = "results/llm_results.csv"

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

RESULTS_PATH = Path(OUTPUT_CSV)


# -----------------------------------------------------------------------
# 2. OPENAI CLIENT
# -----------------------------------------------------------------------

client = OpenAI()


# -----------------------------------------------------------------------
# 3. RETRY MECHANISM FOR OPENAI CALLS
# -----------------------------------------------------------------------

@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
def call_llm(prompt: str) -> str:
    """
    Sends prompt to OpenAI with automatic retries.
    Raises RetryError if still failing after retries.
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    # new SDK: message content is a list of parts; join text parts
    content = response.choices[0].message.content
    if isinstance(content, list):
        text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    else:
        text = content
    return text


# -----------------------------------------------------------------------
# 4. HELPER: AUTO-DETECT START INDEX & PRELOAD OLD RESULTS
# -----------------------------------------------------------------------

def load_existing_results():
    """
    Əgər əvvəldən llm_results.csv varsa:
      - onu oxuyur
      - results listinə çevirir
      - start_index = max(index) + 1 qaytarır
    Yoxdursa:
      - boş results və start_index = 0 qaytarır.
    """
    if RESULTS_PATH.exists():
        prev = pd.read_csv(RESULTS_PATH)
        if "index" in prev.columns and len(prev) > 0:
            start_index = int(prev["index"].max()) + 1
            print(f"🔁 Existing results found. Will continue from index {start_index}.")
            return prev.to_dict("records"), start_index
    print("🆕 No previous results found. Starting from index 0.")
    return [], 0


# -----------------------------------------------------------------------
# 5. MAIN LLM PROCESSING LOOP
# -----------------------------------------------------------------------

def run_llm():
    print("🔄 Loading dataset...")
    df = pd.read_csv(INPUT_CSV)

    # load previous results if exist
    results, start_index = load_existing_results()

    print("🚀 Starting LLM inference...")
    for i in tqdm(range(start_index, len(df))):
        text = df.loc[i, "Text"]  # diqqət: səndə sütun 'Text' idi

        prompt = build_prompt(text, LABELS)

        try:
            try:
                raw_output = call_llm(prompt)
            except RetryError as e:
                # çox güman RateLimitError və ya davamlı network error
                print(f"⚠️ RetryError at row {i}: {e}")
                results.append({
                    "index": i,
                    "text": text,
                    "extracted": "",
                    "reasoning": "",
                    "predicted_label": "RATE_LIMIT_ERROR",
                })
                # çox yüklənməmək üçün bir az böyük pauza ver
                time.sleep(30)
                continue

            # JSON parse
            try:
                parsed = json.loads(raw_output)
            except json.JSONDecodeError:
                # JSON formatında olmayan cavab
                results.append({
                    "index": i,
                    "text": text,
                    "extracted": "",
                    "reasoning": "",
                    "predicted_label": "PARSE_ERROR",
                })
                continue

            results.append({
                "index": i,
                "text": text,
                "extracted": parsed.get("extracted", ""),
                "reasoning": parsed.get("reasoning", ""),
                "predicted_label": parsed.get("label", ""),
            })

        except Exception as e:
            print(f"⚠️ Unexpected error at row {i}: {e}")
            results.append({
                "index": i,
                "text": text,
                "extracted": "",
                "reasoning": "",
                "predicted_label": "ERROR",
            })

        # hər 50 sətirdən bir save
        if i > 0 and i % 50 == 0:
            temp_df = pd.DataFrame(results)
            RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            temp_df.to_csv(RESULTS_PATH, index=False)
            print(f"💾 Progress saved at row {i}")

        # rate-limit riskini azaltmaq üçün
        time.sleep(1.0)

    final_df = pd.DataFrame(results)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(RESULTS_PATH, index=False)
    print("🎉 All done! Saved to results/llm_results.csv")


# -----------------------------------------------------------------------
# 6. RUN
# -----------------------------------------------------------------------

if __name__ == "__main__":
    run_llm()
