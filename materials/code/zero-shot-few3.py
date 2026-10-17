import re
import torch
from typing import List, Dict
from transformers import AutoTokenizer, pipeline

# =====================
# Config
# =====================
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.6
TOP_P = 0.9

# =====================
# Example papers to predict
# =====================
PAPERS: List[Dict[str, str]] = [
    {
        "title": "Neural Approaches to Subject Analysis in Large Digital Libraries",
        "abstract": "We explore transformer-based models to assign Library of Congress Subject Headings (LCSH) using titles and abstracts. We evaluate zero-shot and few-shot settings across scientific domains."
    },
    {
        "title": "A Survey of Microgrid Control with Renewable Energy Integration",
        "abstract": "This study reviews hierarchical control strategies for microgrids integrating PV, BESS, and EVs, focusing on voltage stability, reactive power support, and carbon-aware dispatch."
    }
]

# =====================
# Task instructions
# =====================
base_messages = [
    {"role": "system", "content": "You are a helpful assistant predicting Library of Congress Subject Headings (LCSH) for scientific papers."},
    {"role": "user", "content": "Given the title and abstract of a scientific paper, predict as many relevant LCSH labels as possible."},
    {"role": "user", "content": "Respond only with the predicted LCSH labels separated by commas."}
]

# =====================
# Few-shot examples
# =====================
few_shot_examples = [
    {"role": "user", "content": "Title: The Pentagon Papers: National Security Versus the Public's Right to Know\n"
                                "Abstract: Discusses the Supreme Court trial which resulted from the decision of the New York Times newspaper to publish secret government documents about the Vietnam War."},
    {"role": "assistant", "content": "National security, Freedom of the press, Vietnam War"},

    {"role": "user", "content": "Title: On the Water: Rowing, Yachting, Canoeing, and Lots, Lots, More\n"
                                "Abstract: Describes the sailing, rowing, and canoeing events of the Olympic Games and previews the athletic competition at the 2000 Summer Olympics in Sydney, Australia."},
    {"role": "assistant", "content": "Aquatic sports, Olympics"},

    {"role": "user", "content": "Title: The Andersonville Prison Civil War Crimes Trial: A Headline Court Case\n"
                                "Abstract: Examines the war crimes trial, in which Henry Wirz, the Confederate officer in charge of Andersonville prison camp, was accused of allowing the prisoners to be deliberately abused and neglected."},
    {"role": "assistant", "content": "Andersonville Prison, Civil War, War crimes, Trials, Wirz, Henry, Confederate States of America, Prisoners of war, Human rights abuse"},
]

# =====================
# Helper: clean output
# =====================
def clean_prediction(text: str) -> str:
    text = text.replace("LCSH:", "").replace("Assistant:", "").strip()
    text = text.split("\n")[0]  # take only the first line
    return text

# =====================
# Main
# =====================
def main():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    pipe = pipeline(
        "text-generation",
        model=MODEL_ID,
        tokenizer=tokenizer,
        device_map="auto",
        model_kwargs={"torch_dtype": torch.bfloat16} if torch.cuda.is_available() else {},
        pad_token_id=tokenizer.pad_token_id,
        return_full_text=False
    )

    # Build prompts and run predictions
    for paper in PAPERS:
        msgs = base_messages + few_shot_examples + [
            {"role": "user", "content": f"Title: {paper['title']}\nAbstract: {paper['abstract']}"},
            {"role": "assistant", "content": ""}  # force model to complete as assistant
        ]

        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        output = pipe(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P
        )[0]["generated_text"]

        prediction = clean_prediction(output)

        # Print results
        print("=" * 80)
        print(f"Title   : {paper['title']}")
        print(f"Abstract: {paper['abstract']}")
        print(f"Predicted LCSH: {prediction}")

if __name__ == "__main__":
    main()
