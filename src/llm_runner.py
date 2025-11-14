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

MODEL_NAME = "gpt-4o-mini"  # İstəsən yalnız bu skript üçün gpt-4o yaza bilərsən
INPUT_CSV = "data/merged.csv"
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

HARD_LABELS = [
    "Magnification",
    "Mental filter",
    "Should statements",
    "No Distortion",
]

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
    print("🔄 Loading gold dataset & previous LLM results...")
    df = pd.read_csv(INPUT_CSV)            # columns: Text, Label, source
    res = pd.read_csv(INPUT_RESULTS)       # columns: index, text, extracted, reasoning, predicted_label, ...

    # Gold label-ları index-ə bağlayaq
    gold = df.reset_index().rename(columns={"index": "index", "Label": "gold_label"})

    merged = res.merge(gold[["index", "gold_label"]], on="index", how="left")

    # Rerun edəcəyimiz sətirlər:
    #  - gold label HARD_LABELS siyahısına düşənlər
    #  - və ya predicted_label ERROR tipində olanlar
    mask_hard = merged["gold_label"].isin(HARD_LABELS)
    mask_error = merged["predicted_label"].isin(ERROR_LABELS)

    to_rerun_indices = sorted(merged[mask_hard | mask_error]["index"].unique())
    print(f"🔍 Rows to re-run: {len(to_rerun_indices)}")

    if not to_rerun_indices:
        print("✅ No rows selected for re-run. Exiting.")
        return

    # Nəticələri index-ə görə rahat update etmək üçün
    res = res.set_index("index")

    for i in tqdm(to_rerun_indices):
        try:
            text = df.loc[i, "Text"]
        except KeyError:
            print(f"⚠️ Warning: index {i} not found in merged.csv, skipping.")
            continue

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
                # JSON parse alınmadı
                print(f"⚠️ JSON parse error at row {i}")
                res.loc[i, "extracted"] = ""
                res.loc[i, "reasoning"] = ""
                res.loc[i, "predicted_label"] = "PARSE_ERROR"
                continue

            # Uğurlu nəticə
            res.loc[i, "text"] = text
            res.loc[i, "extracted"] = parsed.get("extracted", "")
            res.loc[i, "reasoning"] = parsed.get("reasoning", "")
            res.loc[i, "predicted_label"] = parsed.get("label", "")

        except Exception as e:
            print(f"⚠️ Unexpected error at row {i}: {e}")
            res.loc[i, "extracted"] = ""
            res.loc[i, "reasoning"] = ""
            res.loc[i, "predicted_label"] = "ERROR"

        # Hər 50 sətirdən bir intermediate save
        if i % 50 == 0:
            out_path = Path(OUTPUT_RESULTS)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            res.reset_index().to_csv(out_path, index=False)
            print(f"💾 Partial save after index {i}")

        time.sleep(1.0)  # rate-limit üçün kiçik pauza

    # Final save
    out_path = Path(OUTPUT_RESULTS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.reset_index().to_csv(out_path, index=False)
    print(f"🎉 Done. Saved updated results to {OUTPUT_RESULTS}")


if __name__ == "__main__":
    main()
