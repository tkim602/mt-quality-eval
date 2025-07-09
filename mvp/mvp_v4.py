#!/usr/bin/env python3
import os
import json
import gzip
import re
import sys
from pathlib import Path
from math import sqrt
from difflib import SequenceMatcher
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from tqdm import tqdm
import openai
from openai.error import APIConnectionError, Timeout
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_CHAT     = os.getenv("MODEL_CHAT", "gpt-3.5-turbo")
BATCH_SIZE     = int(os.getenv("BATCH_SIZE", "32"))
THRESH_SIM     = float(os.getenv("THRESH_SIM", "0.8"))
THRESH_REUSE   = float(os.getenv("THRESH_REUSE", "0.9"))

INPUT_DIR      = Path("samples")
OUTPUT_PATH    = Path("out_gpt3.5_tb_KoE5.ndjson.gz")
CACHE_PATH     = Path(".cache.json")
TERM_MAP_PATH  = INPUT_DIR / "term_base_map.json"
REUSE_TM_PATH  = INPUT_DIR / "tm.json"

openai.api_key = OPENAI_API_KEY

embedding_model = SentenceTransformer("nlpai-lab/KoE5")
tokenizer       = AutoTokenizer.from_pretrained("nlpai-lab/KoE5")

PLACEHOLDER_PATTERN = re.compile(r"\{[^}]+\}")

def ko_morph_tokens(text: str) -> set[str]:
    raw = tokenizer.tokenize(text)
    return set(t.lstrip("##") for t in raw)

def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8") or "{}")

@retry(
    retry=retry_if_exception_type((openai.error.OpenAIError, APIConnectionError, Timeout)),
    wait=wait_exponential(min=2, max=20),
    stop=stop_after_attempt(6),
)
def call_chat(msgs, *, timeout_sec=60):
    return openai.ChatCompletion.create(
        model=MODEL_CHAT,
        messages=msgs,
        temperature=0,
        request_timeout=timeout_sec,
    )

def get_embedding(text: str):
    return embedding_model.encode(text, normalize_embeddings=False).tolist()

def cosine(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    na = sqrt(sum(x*x for x in a))
    nb = sqrt(sum(y*y for y in b))
    return dot/(na*nb) if na and nb else 0.0

def extract_numbers(text):
    return re.findall(r"\d+(?:[.,]\d+)?", text)

def extract_entities(text):
    return re.findall(r"\b[A-Z][A-Za-z0-9_]+\b|\b\w+_\w+\b|\b[A-Z0-9_]{2,}\b", text)

def fuzzy_ratio(a,b):
    return SequenceMatcher(None,a,b).ratio()

def glossary_lines(tb):
    return "\n".join(f"{e} = {k}" for e,k in tb.items())

def missing_terms(src: str, tgt: str, ko2en: dict[str,str]) -> list[str]:
    st = ko_morph_tokens(src)
    kos = [k for k in ko2en if k in st]
    tt = {t.lower() for t in re.findall(r"\w+", tgt)}
    return [ko2en[k] for k in kos if ko2en[k].lower() not in tt]

def back_translate(text: str, gloss: str) -> str:
    if not text.strip() or PLACEHOLDER_PATTERN.fullmatch(text.strip()):
        return text
    msgs = [
        {
            "role": "system",
            "content": (
                "You are an expert Korean technical translator.\n\n"
                "Context\n"
                "• The English sentence you receive was originally translated "
                "from Korean and belongs to a cybersecurity / software UI.\n"
                "• A termbase (glossary) must be obeyed exactly.\n\n"
                "Strict Guidelines\n"
                "1. Apply the glossary without deviation: no synonyms, "
                "inflections, or omissions.\n"
                "2. Preserve placeholders such as {0}, {1}, {name}, etc., exactly.\n"
                "3. Preserve file paths, code identifiers, API names, CVE numbers, etc.\n"
                "4. Do NOT add, remove, or reorder information.\n"
                "5. Use concise, formal Korean typical of professional "
                "software UIs.\n"
                "6. Return Korean text only — no quotation marks or extra lines.\n\n"
                "Glossary (en = ko)\n" + gloss
            ),
        },
        {"role": "user", "content": text},
    ]
    return call_chat(msgs).choices[0].message.content.strip()

def build_eval_prompt():
    return {
        "role": "system",
        "content": (
            "You are a senior localization QA specialist assessing "
            "Korean→English UI strings in the cybersecurity / software domain.\n\n"
            "INPUT – JSON array; each element: {key, source, translation}.\n\n"
            "TASKS\n"
            "1. Score each item on three axes (0-100 integers).\n"
            "   • overall  – holistic quality (≈ min(adequacy, fluency)).\n"
            "   • adequacy – completeness & accuracy vs. source meaning.\n"
            "   • fluency  – naturalness, grammar, target-language conventions.\n\n"
            "2. Rubric\n"
            "   0-25  Unusable or wrong meaning\n"
            "   26-50 Serious issues, meaning partly lost\n"
            "   51-75 Understandable but terminologically / stylistically flawed\n"
            "   76-100 Near-perfect to perfect\n\n"
            "3. Mandatory checks (failures must affect scores & comments)\n"
            "   • All numerals, placeholders, identifiers must appear unchanged.\n"
            "   • Glossary adherence — every Korean term in the glossary MUST "
            "map to the prescribed English term.\n"
            "   • Short UI fragments (≤3 words) may lack verbs; omit fluency "
            "penalties for that.\n\n"
            "4. If overall < 60 provide up to 3 improved English alternatives "
            "in ‘alternatives’; otherwise leave it empty.\n\n"
            "OUTPUT – JSON array (same order) with exactly:\n"
            "key, overall, adequacy, fluency, comments (array), alternatives (array).\n"
            "Return the JSON array only – no prose before or after."
        ),
    }

def score_with_openai(batch):
    msgs = [
        build_eval_prompt(),
        {
            "role": "user",
            "content": json.dumps(
                [{"key":k,"source":s,"translation":t} for k,s,t in batch],
                ensure_ascii=False,
            ),
        },
    ]
    resp = call_chat(msgs)
    try:
        return json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        return [{"key":k,"overall":None,"adequacy":None,"fluency":None,
                 "comments":[],"alternatives":[]} for k,_,_ in batch]

def main():
    cache      = load_json(CACHE_PATH)
    term_pairs = load_json(TERM_MAP_PATH)
    ko2en      = {p["ko"]:p["en-US"] for p in term_pairs}
    en2ko      = {p["en-US"]:p["ko"] for p in term_pairs}
    gloss      = glossary_lines(en2ko)

    zu        = load_json(INPUT_DIR/"zu_checker.json")
    en        = load_json(INPUT_DIR/"en-US_checker.json")
    reuse_tm  = load_json(REUSE_TM_PATH)

    items = [(k, zu[k], en.get(k,"")) for k in list(zu.keys())[:40]]
    results = []

    for i in tqdm(range(0, len(items), BATCH_SIZE), desc="Batches"):
        batch = items[i:i+BATCH_SIZE]
        scored, to_call = [], []
        for k, src, tgt in batch:
            ck = f"{k}:{tgt}"
            if ck in cache:
                scored.append(cache[ck])
            else:
                to_call.append((k, src, tgt))

        if to_call:
            resp = score_with_openai(to_call)
            for r in resp:
                cache[f"{r['key']}:{en.get(r['key'],'')}"] = r
            scored += resp

        for rec in scored:
            k   = rec["key"]
            src = zu[k]
            tgt = en.get(k,"")

            bt  = back_translate(tgt, gloss)
            sim = cosine(get_embedding(src), get_embedding(bt))

            miss_nums = list(set(extract_numbers(src)) - set(extract_numbers(tgt)))
            miss_ents = list(set(extract_entities(src)) - set(extract_entities(tgt)))
            miss_base = missing_terms(src, tgt, ko2en)

            prev    = reuse_tm.get(k,"")
            reuse_r = fuzzy_ratio(prev, tgt) if prev else 0

            fallback = [bt] if (sim < THRESH_SIM or miss_nums or miss_ents or miss_base) else []

            rec.update(
                semantic_sim     = round(sim,3),
                missing_numbers  = miss_nums,
                missing_entities = miss_ents,
                missing_terms    = miss_base,
                reuse_ratio      = round(reuse_r,3),
                fallback         = fallback,
            )
            results.append(rec)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUTPUT_PATH, "wt", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Done →", OUTPUT_PATH)

if __name__ == "__main__":
    main()
