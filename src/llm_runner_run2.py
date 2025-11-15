# src/llm_runner_run2.py

import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

from prompts import build_prompt

# ---------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------

MODEL_NAME = "gpt-4o-mini"  # istəsən "gpt-4o" edə bilərsən
INPUT_CSV = "data/merged.csv"

# Hər yeni inference üçün bunu dəyiş:
RUN_NAME = "llm_results_run2"   # məsələn: llm_results_gpt4o_2025_11_14
OUTPUT_CSV = f"results/{RUN_NAME}.csv"

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

client = OpenAI()


# ---------------------------------------------------
# 2. RETRY LAYER
# ---------------------------------------------------

@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
def call_llm(prompt: str) -> str:
    """
    LLM call with retries. Raises RetryError if still failing.
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = response.choices[0].message.content
    if isinstance(content, list):
        text = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    else:
        text = content
    return text


def load_existing_results(path: Path):
    """
    Əgər OUTPUT_CSV artıq mövcuddursa, onu oxuyur və
    start_index = max(index)+1 qaytarır. Yoxdursa, 0-dan başlayır.
    """
    if path.exists():
        prev = pd.read_csv(path)
        if "index" in prev.columns and len(prev) > 0:
            start_index = int(prev["index"].max()) + 1
            print(f"🔁 Existing results found. Continue from index {start_index}.")
            return prev.to_dict("records"), start_index
    print("🆕 No previous results. Starting from index 0.")
    return [], 0


# ---------------------------------------------------
# 3. MAIN LOOP
# ---------------------------------------------------

def run_llm():
    print("🔄 Loading dataset...")
    df = pd.read_csv(INPUT_CSV)

    # text sütununun adını avtomatik tapmaq
    if "Text" in df.columns:
        text_col = "Text"
    elif "text" in df.columns:
        text_col = "text"
    else:
        raise ValueError(f"Text column not found. Available columns: {df.columns.tolist()}")

    out_path = Path(OUTPUT_CSV)
    results, start_index = load_existing_results(out_path)

    print(f"🚀 Starting LLM inference on {len(df)} rows...")
    for i in tqdm(range(start_index, len(df))):
        text = df.loc[i, text_col]

        prompt = build_prompt(text, LABELS)

        try:
            try:
                raw_output = call_llm(prompt)
            except RetryError as e:
                print(f"⚠️ RetryError at row {i}: {e}")
                results.append({
                    "index": i,
                    "text": text,
                    "extracted": "",
                    "reasoning": "",
                    "predicted_label": "RATE_LIMIT_ERROR",
                })
                time.sleep(30)
                continue

            # JSON parse
            try:
                parsed = json.loads(raw_output)
            except json.JSONDecodeError:
                print(f"⚠️ JSON parse error at row {i}")
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

        # hər 50 sətrdən bir save
        if i > 0 and i % 50 == 0:
            tmp_df = pd.DataFrame(results)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_df.to_csv(out_path, index=False)
            print(f"💾 Progress saved at row {i} -> {OUTPUT_CSV}")

        time.sleep(1.0)  # rate-limit riskini azaltmaq üçün

    final_df = pd.DataFrame(results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)
    print(f"🎉 All done! Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    run_llm()
