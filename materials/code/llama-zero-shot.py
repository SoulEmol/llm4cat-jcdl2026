import re
import torch
from typing import List, Dict
from transformers import AutoTokenizer, pipeline

# ============ Config ============
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
BATCH_SIZE = 4
MAX_NEW_TOKENS = 160
TEMPERATURE = 0.6
TOP_P = 0.9

# Directly provide the papers here (you can add more items)
PAPERS: List[Dict[str, str]] = [
    {
        "title": "Neural Approaches to Subject Analysis in Large Digital Libraries",
        "abstract": "We explore transformer-based models to assign Library of Congress Subject Headings (LCSH) using titles and abstracts. We evaluate zero-shot and few-shot settings across scientific domains."
    },
    {
        "title": "A Survey of Microgrid Control with Renewable Energy Integration",
        "abstract": "This study reviews hierarchical control strategies for microgrids integrating PV, BESS, and EVs, focusing on voltage stability, reactive power support, and carbon-aware dispatch."
    },
]
# ===============================

# System and user instructions
BASE_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant predicting Library of Congress Subject Headings (LCSH) for scientific papers."},
    {"role": "user", "content": "Given the title and abstract of a book or paper, predict as many relevant LCSH labels as possible."},
    {"role": "user", "content": "Respond only with the predicted LCSH labels separated by commas."}
]

def clean_to_commas_only(text: str) -> str:
    """Clean the raw model output so it only contains comma-separated labels."""
    text = re.sub(r"^```.*?\n|\n```$", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^\s*(?:LCSH\s*:|Labels\s*:)\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[\n;]+", ", ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,.\n\t")
    return text

def main():
    # Load tokenizer and pipeline
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    textgen = pipeline(
        "text-generation",
        model=MODEL_ID,
        tokenizer=tokenizer,
        device_map="auto",  # automatically selects available GPU(s)
        model_kwargs={"torch_dtype": torch.bfloat16} if torch.cuda.is_available() else {},
        return_full_text=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Build prompts using the Llama chat template
    def make_prompt(title: str, abstract: str) -> str:
        msgs = BASE_MESSAGES + [{"role": "user", "content": f"Title: {title}\nAbstract: {abstract}"}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    prompts = [make_prompt(p["title"], p["abstract"]) for p in PAPERS]

    # Run inference
    preds: List[str] = []
    for i in range(0, len(prompts), BATCH_SIZE):
        chunk = prompts[i:i + BATCH_SIZE]
        outs = textgen(
            chunk,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,           # for stochastic output; set to False for deterministic
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        for o in outs:
            gen = o.get("generated_text", "") or ""
            preds.append(clean_to_commas_only(gen) if gen else "Error")

    # Print results
    for paper, pred in zip(PAPERS, preds):
        print("=" * 80)
        print(f"Title   : {paper['title']}")
        print(f"Abstract: {paper['abstract']}")
        print(f"Predicted LCSH: {pred}")

if __name__ == "__main__":
    main()
