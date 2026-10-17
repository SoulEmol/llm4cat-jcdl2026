import os
import re
from typing import List, Dict
from transformers import pipeline

# Configure GPU (optional)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(f"Using device: {os.environ.get('CUDA_VISIBLE_DEVICES', 'cpu')}")

# Load Llama 3.1 8B-Instruct model
model_id = "meta-llama/Llama-3.1-8B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": "bfloat16"},
    device=0,
    pad_token_id=0
)
pipe.tokenizer.pad_token = pipe.tokenizer.eos_token

# ---------------------------
# Manually provided items
# Each item: title, abstract, Predicted_Label_Count (n)
# ---------------------------
ITEMS: List[Dict[str, str]] = [
    {
        "title": "Neural Approaches to Subject Analysis in Large Digital Libraries",
        "abstract": "We explore transformer-based models to assign Library of Congress Subject Headings (LCSH) using titles and abstracts. We evaluate zero-shot and few-shot settings across scientific domains.",
        "Predicted_Label_Count": 6
    },
    {
        "title": "A Survey of Microgrid Control with Renewable Energy Integration",
        "abstract": "This study reviews hierarchical control strategies for microgrids integrating PV, BESS, and EVs, focusing on voltage stability, reactive power support, and carbon-aware dispatch.",
        "Predicted_Label_Count": 5
    },
]

# ---------------------------
# Helpers (same cleaning logic as your original)
# ---------------------------
def clean_lcsh_labels(label_str):
    """
    Clean LCSH predictions:
    - Remove content after `--` or `-`
    - Remove newlines, dots, etc.
    - Remove duplicates and sort
    """
    if isinstance(label_str, list):
        label_str = ", ".join(map(str, label_str))

    if not isinstance(label_str, str) or label_str.strip() == "":
        return ""

    labels = label_str.split(",")
    cleaned_labels = set()

    for label in labels:
        label = label.strip()
        label = re.sub(r"\s*--.*", "", label)
        label = re.sub(r"\s+-\s+.*", "", label)
        if label:
            cleaned_labels.add(label)

    # keep sorted for determinism
    return ", ".join(sorted(cleaned_labels))

def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Convert your original `messages` (list of {role, content}) into a plain prompt string
    so it works with pipeline('text-generation').
    """
    parts = []
    for m in messages:
        role = m.get("role", "user").capitalize()
        content = m.get("content", "")
        parts.append(f"{role}: {content}")
    # Add a final "Assistant:" to cue completion
    return "\n\n".join(parts) + "\n\nAssistant: "

def generate_lcsh_labels(title, abstract, n):
    """
    Step-by-step LCSH prediction:
    - Round 1: generate n labels
    - Round 2: generate 2n more labels
    - Round 3: generate as many additional labels as possible
    """
    def generate_step(messages):
        # Convert your message list to a single text prompt
        prompt = _messages_to_prompt(messages)

        response = pipe(
            prompt,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_full_text=False
        )

        if isinstance(response, list) and len(response) > 0 and "generated_text" in response[0]:
            generated_text = response[0]["generated_text"]
            generated_text = (generated_text or "").strip()

            # Extract everything after the last instruction sentence if present
            # (robust fallback: just use the whole completion)
            anchor = "Respond only with the predicted LCSH labels separated by commas."
            if anchor in prompt:
                # nothing needed; we already limited format in instruction
                pass

            labels = [label.strip() for label in generated_text.split(",") if label.strip()]
            labels = clean_lcsh_labels(labels)
            return labels

        print("Warning: unexpected response format:", response)
        return ""

    # Round 1
    r1 = generate_step([
        {"role": "user", "content": f"Title: {title}\nAbstract: {abstract}"},
        {"role": "user", "content": f"Predict exactly {n} Library of Congress Subject Headings (LCSH) labels. Respond only with the predicted LCSH labels separated by commas."}
    ])

    # Round 2
    r2 = ""
    if r1:
        r2 = generate_step([
            {"role": "user", "content": f"Title: {title}\nAbstract: {abstract}"},
            {"role": "user", "content": f"Current LCSH labels: {r1}"},
            {"role": "user", "content": f"Predict {2*n} additional Library of Congress Subject Headings (LCSH) labels. Respond only with the predicted LCSH labels separated by commas."}
        ])

    # Round 3
    r3 = ""
    if r2:
        r3 = generate_step([
            {"role": "user", "content": f"Title: {title}\nAbstract: {abstract}"},
            {"role": "user", "content": f"Current LCSH labels: {r2}"},
            {"role": "user", "content": f"Predict as many additional Library of Congress Subject Headings (LCSH) labels as possible. Respond only with the predicted LCSH labels separated by commas."}
        ])

    # Merge & deduplicate
    all_labels = set(filter(None, [
        *r1.split(", ") if r1 else [],
        *r2.split(", ") if r2 else [],
        *r3.split(", ") if r3 else []
    ]))

    return clean_lcsh_labels(", ".join(all_labels))

# ---------------------------
# Run and print
# ---------------------------
def main():
    print("Starting manual prediction...")
    for i, it in enumerate(ITEMS, 1):
        title = it["title"]
        abstract = it["abstract"]
        n = int(it["Predicted_Label_Count"])
        merged = generate_lcsh_labels(title, abstract, n)

        print("=" * 100)
        print(f"[{i}] Title   : {title}")
        print(f"[{i}] Abstract: {abstract}")
        print(f"[{i}] n       : {n}")
        print(f"[{i}] Predicted LCSH: {merged}")

if __name__ == "__main__":
    main()

