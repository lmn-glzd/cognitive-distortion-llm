# src/prompts.py

def build_prompt(text, label_options):
    """
    ERD-style structured reasoning prompt generator.
    text: input sentence from dataset
    label_options: list of possible distortion labels
    """

    labels_str = "\n".join([f"- {lbl}" for lbl in label_options])

    prompt = f"""
You are an expert cognitive distortion analyst. 
Your task is to classify a given sentence into ONE cognitive distortion type.
Use structured reasoning similar to ERD (extract → reason → decide).

### INPUT STATEMENT:
"{text}"

### POSSIBLE DISTORTION TYPES:
{labels_str}

### FOLLOW THIS FORMAT STRICTLY:
1. Extract key phrases from the statement.
2. Explain the reasoning step-by-step.
3. Decide on ONE final distortion label from the list.

### OUTPUT FORMAT (JSON):
{{
  "extracted": "...",
  "reasoning": "...",
  "label": "one_of_the_labels_above"
}}
"""

    return prompt
