# src/llm_rerun_hard_cases.py

import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

from prompts import build_prompt

# ---------------------------------------------
# 1. CONFIG
# ---------------------------------------------

MODEL_NAME = "gpt-4o-mini"
INPUT_RESULTS = "results/llm_results.csv"
OUTPUT_RESULTS = "results/llm_results_v2.csv"

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

# Bunlar LLM üçün xüsusilə çətin hesab etdiyimiz label-lərdir
HARD_LABELS = [
    "Magnification",
    "Mental filter",
    "Should statements",
    "No Distortion",
]

# Bu label-lər texniki səhvləri göstərir
ERROR_LABELS = ["ERROR", "PARSE_ERROR", "RATE_LIMIT_ERROR"]

client = OpenAI()


# ---------------------------------------------
# 2. RETRY LAYER
# ---------------------------------------------

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


# ---------------------------------------------
# 3. MAIN
# ---------------------------------------------

def main():
    print("🔄 Loading previous LLM results...")
    res = pd.read_csv(INPUT_RESULTS)

    if "index" not in res.columns:
        raise ValueError("llm_results.csv must contain an 'index' column.")

    # Sətirləri seçirik:
    #  - predicted_label ERROR tipində olanlar
    #  - predicted_label HARD_LABELS siyahısına düşənlər
    mask_error = res["predicted_label"].isin(ERROR_LABELS)
    mask_hard_pred = res["predicted_label"].isin(HARD_LABELS)

    to_rerun_ids = sorted(res.loc[mask_error | mask_hard_pred, "index"].unique())
    print(f"🔍 Rows to re-run (by 'index' column): {len(to_rerun_ids)}")

    if not to_rerun_ids:
        print("✅ No rows selected for re-run. Exiting.")
        return

    # 'index' sütununu DataFrame index-ə çeviririk ki, rahat update edək
    res = res.set_index("index")

    for n, i in enumerate(tqdm(to_rerun_ids, desc="Re-running hard/error cases"), start=1):
        if i not in res.index:
            print(f"⚠️ Warning: index {i} not found in llm_results DataFrame, skipping.")
            continue

        text = res.loc[i, "text"]

        prompt = build_prompt(text, LABELS)

        try:
            try:
                raw_output = call_llm(prompt)
            except RetryError as e:
                print(f"⚠️ RetryError at row {i}: {e}")
                res.loc[i, "extracted"] = ""
                res.loc[i, "reasoning"] = ""
                res.loc[i, "predicted_label"] = "RATE_LIMIT_ERROR"
                time.sleep(30)
                continue

            try:
                parsed = json.loads(raw_output)
            except json.JSONDecodeError:
                print(f"⚠️ JSON parse error at row {i}")
                res.loc[i, "extracted"] = ""
                res.loc[i, "reasoning"] = ""
                res.loc[i, "predicted_label"] = "PARSE_ERROR"
                continue

            # Uğurlu nəticə
            res.loc[i, "extracted"] = parsed.get("extracted", "")
            res.loc[i, "reasoning"] = parsed.get("reasoning", "")
            res.loc[i, "predicted_label"] = parsed.get("label", "")

        except Exception as e:
            print(f"⚠️ Unexpected error at row {i}: {e}")
            res.loc[i, "extracted"] = ""
            res.loc[i, "reasoning"] = ""
            res.loc[i, "predicted_label"] = "ERROR"

        # Hər 50 sətirdən bir intermediate save
        if n % 50 == 0:
            out_path = Path(OUTPUT_RESULTS)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            res.reset_index().to_csv(out_path, index=False)
            print(f"💾 Partial save after {n} rows (up to index {i})")

        time.sleep(1.0)  # rate-limit üçün kiçik pauza

    # Final save
    out_path = Path(OUTPUT_RESULTS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.reset_index().to_csv(out_path, index=False)
    print(f"🎉 Done. Saved updated results to {OUTPUT_RESULTS}")


if __name__ == "__main__":
    main()
