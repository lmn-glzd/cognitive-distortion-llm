# src/prompts.py

from textwrap import dedent

def build_prompt(text: str, labels: list[str]) -> str:
    """
    Gücləndirilmiş ERD-style prompt:
    - Hər label üçün qısa tərif + example
    - No Distortion üçün sərt qayda
    - Sonda yalnız valid JSON qaytarma tələbi
    """

    # Label tərifləri (datasetdəki 11 sinifə uyğun)
    definitions = {
        "All-or-Nothing Thinking": dedent("""
            Thinking in extreme, black-and-white categories such as "always / never", "completely / totally", 
            without recognizing shades of gray.
            Example: "If I’m not perfect, I’m a complete failure."
        """).strip(),

        "Emotional Reasoning": dedent("""
            Treating emotions as evidence for facts – assuming that because I feel something, it must be true.
            Example: "I feel worthless, so I must be worthless."
        """).strip(),

        "Fortune-telling": dedent("""
            Predicting the future in a negative way without sufficient evidence.
            Example: "The party will definitely be a disaster; no one will like me."
        """).strip(),

        "Labeling": dedent("""
            Attaching a global, negative label to yourself or others based on a single event or behavior.
            Example: "I forgot one thing; I’m such an idiot."
        """).strip(),

        "Magnification": dedent("""
            Exaggerating the importance or severity of problems, mistakes, or threats ("blowing things out of proportion").
            Example: "If I make this mistake, my whole life will be ruined."
        """).strip(),

        "Mental filter": dedent("""
            Focusing almost exclusively on one negative detail while ignoring the broader, more balanced picture.
            Example: "I got many compliments, but I only think about the one criticism."
        """).strip(),

        "Mind Reading": dedent("""
            Assuming you know what others are thinking (usually something negative) without clear evidence.
            Example: "She didn’t text back; she must think I’m annoying."
        """).strip(),

        "No Distortion": dedent("""
            The thought is generally realistic, balanced, and does not clearly fit any of the above distortion types.
            There may be emotion, but no clear cognitive error as defined above.
        """).strip(),

        "Overgeneralization": dedent("""
            Drawing a broad, sweeping conclusion from a single event ("always", "never", "everything", "nothing").
            Example: "I failed this exam, so I will always fail at everything."
        """).strip(),

        "Personalization": dedent("""
            Taking excessive responsibility or blame for events outside your full control, 
            or assuming that everything is about you.
            Example: "My friend is upset; it must be my fault."
        """).strip(),

        "Should statements": dedent("""
            Using rigid 'should', 'must', or 'have to' rules for yourself or others, often with self-criticism or guilt.
            Example: "I should never feel anxious. I must always be in control."
        """).strip(),
    }

    # Definitions hissəsini string kimi yığaq
    label_descriptions = []
    for label in labels:
        desc = definitions.get(label, "").strip()
        label_descriptions.append(f"- **{label}**: {desc}")
    labels_block = "\n\n".join(label_descriptions)

    allowed_labels_str = ", ".join(f'"{lbl}"' for lbl in labels)

    prompt = f"""
    You are an expert CBT (Cognitive Behavioral Therapy) clinician who specializes in
    identifying cognitive distortions in short text thoughts.

    Your task: 
    1. Read the given thought.
    2. Decide whether it contains a cognitive distortion.
    3. If yes, choose the **single best fitting** distortion label from the list below.
    4. If the thought is generally realistic and does not clearly fit any distortion definition,
       choose "No Distortion".

    Available labels and definitions:

    {labels_block}

    IMPORTANT RULES:
    - Choose exactly ONE label.
    - The chosen label MUST be one of: {allowed_labels_str}.
    - Only choose "No Distortion" if none of the other labels clearly apply.
    - If multiple distortions could apply, choose the one that best captures the main thinking error.

    Now analyze the following thought:

    THOUGHT:
    \"\"\"{text}\"\"\"

    Respond with **ONLY** a valid JSON object, with this exact structure:

    {{
      "extracted": "the key part(s) of the thought that show the distortion (or the whole thought)",
      "reasoning": "a brief explanation of why you chose this label, referencing the definitions above",
      "label": "one of the allowed labels, exactly as written"
    }}

    Do not include any additional text, comments, or formatting outside of the JSON.
    """

    return dedent(prompt).strip()
