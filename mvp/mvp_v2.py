#!/usr/bin/env python3
import os
import json
import gzip
import re
from pathlib import Path
from dotenv import load_dotenv
import openai
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from tqdm import tqdm
from difflib import SequenceMatcher
from math import sqrt

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_CHAT = os.getenv("MODEL_CHAT", "gpt-4.1")
MODEL_EMB = os.getenv("MODEL_EMB", "text-embedding-3-small")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))
INPUT_DIR = Path("samples")
OUTPUT_PATH = Path("out_gpt4.1_checker.ndjson.gz")
CACHE_PATH = Path(".cache.json")
TERM_BASE_PATH = INPUT_DIR / "term_base.json"
REUSE_TM_PATH = INPUT_DIR / "tm.json"
THRESH_SIM = float(os.getenv("THRESH_SIM", "0.8"))
THRESH_REUSE = float(os.getenv("THRESH_REUSE", "0.9"))

openai.api_key = OPENAI_API_KEY

def load_cache(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8") or "{}")

cache = load_cache(CACHE_PATH)
term_base = set(json.loads(TERM_BASE_PATH.read_text(encoding="utf-8")))
reuse_tm = json.loads(REUSE_TM_PATH.read_text(encoding="utf-8"))

@retry(
    retry=retry_if_exception_type(openai.error.OpenAIError),
    wait=wait_exponential(min=1, max=10),
    stop=stop_after_attempt(5)
)
def call_chat(messages):
    return openai.ChatCompletion.create(model=MODEL_CHAT, messages=messages, temperature=0)

@retry(
    retry=retry_if_exception_type(openai.error.OpenAIError),
    wait=wait_exponential(min=1, max=10),
    stop=stop_after_attempt(5)
)
def get_embedding(text):
    resp = openai.Embedding.create(model=MODEL_EMB, input=text)
    return resp.data[0].embedding

def cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = sqrt(sum(x*x for x in a))
    nb = sqrt(sum(y*y for y in b))
    return dot/(na*nb) if na and nb else 0.0

def extract_numbers(text):
    return re.findall(r"\d+(?:[.,]\d+)?", text)

def extract_entities(text):
    return re.findall(r"\b[A-Z][a-zA-Z0-9_]+\b", text)

def fuzzy_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()

def build_system_msg():
    return {
        "role": "system",
        "content": (
        "You are a professional translation quality analyst specializing in Korean→English UI translations for software interfaces.\n"
        "\n"
        "For each input item (an object with “key”, “source” [Korean], and “translation” [English]):\n"
        "Each input item pertains to software vulnerabilities and may include technical terminology specific to programming languages, security concepts, or libraries.\n"
        "Therefore, when validating these inputs, it is important to preserve the accuracy, semantics, and context of these technical phrases.\n"
        "\n"
        "1. Evaluate three scores (integers 0–100):\n"
        "   • overall: holistic judgment combining adequacy and fluency\n"
        "   • adequacy: how fully and accurately the English conveys the Korean meaning\n"
        "   • fluency: how natural, grammatical, and context-appropriate the English is\n"
        "\n"
        "2. Use this 4-tier scale:\n"
        "   0–25    Incomprehensible or wrong meaning\n"
        "   26–50   Partial meaning with serious issues\n"
        "   51–75   Mostly correct meaning, some awkward phrasing\n"
        "   76–100  Accurate meaning with native-level fluency\n"
        "\n"
        "3. Special rules:\n"
        "   • Short UI phrases (1–3 words: button labels, menu items, column headers) may be fragments; do not penalize fluency if they clearly express intent.\n"
        "   • All numeric values, technical terms, API names, function names, and other proper nouns present in the source MUST appear unchanged in the translation.\n"
        "\n"
        "4. Output requirements:\n"
        "   • Return a JSON array of objects, in the same order as the input.\n"
        "   • Each object must contain exactly these keys:\n"
        "       – key: string\n"
        "       – overall: integer 0–100\n"
        "       – adequacy: integer 0–100\n"
        "       – fluency: integer 0–100\n"
        "       – comments: array of strings (concise feedback points)\n"
        "       – alternatives: array of up to 3 improved English renditions, ONLY if overall < 60; otherwise an empty array\n"
        "   • Do NOT include any additional fields or explanatory text.\n"
        "\n"
        "Respond strictly with the JSON array—no prose or commentary."
        )
    }

def score_with_openai(batch):
    msgs = [
        build_system_msg(),
        {
            "role": "user",
            "content": json.dumps(
                [{"key": k, "source": src, "translation": tgt} for k, src, tgt in batch],
                ensure_ascii=False
            )
        }
    ]
    resp = call_chat(msgs)
    try:
        return json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        out = []
        for k, _, _ in batch:
            out.append({
                "key": k,
                "overall": None,
                "adequacy": None,
                "fluency": None,
                "comments": [resp.choices[0].message.content],
                "alternatives": []
            })
        return out

def back_translate(text):
    msgs = [
        {"role": "system", "content": "Translate the following English back to Korean."},
        {"role": "user", "content": text}
    ]
    resp = call_chat(msgs)
    return resp.choices[0].message.content.strip()

def main():
    zu = json.loads((INPUT_DIR / "zu_checker.json").read_text(encoding="utf-8"))
    en = json.loads((INPUT_DIR / "en-US_checker.json").read_text(encoding="utf-8"))
    items = [(k, zu[k], en.get(k, "")) for k in list(zu.keys())[:100]]
    results = []

    for i in tqdm(range(0, len(items), BATCH_SIZE), desc="Batches"):
        batch = items[i:i+BATCH_SIZE]
        scored = []
        to_call = []

        for k, src, tgt in batch:
            ck = f"{k}:{tgt}"
            if ck in cache:
                scored.append(cache[ck])
            else:
                to_call.append((k, src, tgt))

        if to_call:
            resp = score_with_openai(to_call)
            for rec in resp:
                cache[f"{rec['key']}:{en[rec['key']]}"] = rec
            scored += resp

        for rec in scored:
            k = rec["key"]
            src = zu[k]
            tgt = en.get(k, "")
            bt = back_translate(tgt)
            emb_src = get_embedding(src)
            emb_bt = get_embedding(bt)
            sim = cosine(emb_src, emb_bt)
            nums_src = extract_numbers(src)
            nums_tgt = extract_numbers(tgt)
            missing_nums = list(set(nums_src) - set(nums_tgt))
            ents_src = extract_entities(src)
            ents_tgt = extract_entities(tgt)
            missing_ents = list(set(ents_src) - set(ents_tgt))
            tokens = set(re.findall(r"\w+", src))
            missing_terms = [t for t in term_base if t not in tokens]
            reuse_ratio = 0
            prev = reuse_tm.get(k, "")
            if prev:
                reuse_ratio = fuzzy_ratio(prev, tgt)
            fallback_suggestion = []
            if sim < THRESH_SIM or missing_nums or missing_ents:
                fallback_suggestion = [back_translate(tgt)]
            rec.update({
                "semantic_sim": round(sim, 3),
                "missing_numbers": missing_nums,
                "missing_entities": missing_ents,
                "missing_terms": missing_terms,
                "reuse_ratio": round(reuse_ratio, 3),
                "fallback": fallback_suggestion
            })
            results.append(rec)

    with gzip.open(OUTPUT_PATH, "wt", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Done. Output:", OUTPUT_PATH)

if __name__ == "__main__":
    main()
