#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import os
import re
from difflib import SequenceMatcher
from math import sqrt
from pathlib import Path
from typing import Any, Dict, List, Tuple

import openai
from dotenv import load_dotenv
from openai.error import APIConnectionError, OpenAIError, Timeout
from sentence_transformers import SentenceTransformer
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm
from transformers import AutoTokenizer

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL_CHAT = os.getenv("MODEL_CHAT", "gpt-3.5-turbo")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))
THRESH_SIM = float(os.getenv("THRESH_SIM", "0.80"))
THRESH_REUSE = float(os.getenv("THRESH_REUSE", "0.90"))

INPUT_DIR = Path("samples")
OUTPUT_DIR = Path("out"); OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "out_optimized.ndjson.gz"
CACHE_PATH = Path(".cache.json")

TB_JSON = INPUT_DIR / "term_base_map.json"
REUSE_TM_PATH = INPUT_DIR / "tm.json"
KO_JSON = INPUT_DIR / "ko_checker.json"
EN_JSON = INPUT_DIR / "en-US_checker.json"

print("loading KoE5 … first run may take a while")
EMBED_MODEL = SentenceTransformer("nlpai-lab/KoE5")
TOKENIZER = AutoTokenizer.from_pretrained("nlpai-lab/KoE5")
PLACEHOLDER_PATTERN = re.compile(r"\{[^}]+\}")

def ko_tokens(text: str) -> set[str]:
    return {t.lstrip("##") for t in TOKENIZER.tokenize(text)}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8") or "{}")


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sqrt(sum(x * x for x in a)); nb = sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def extract_numbers(t: str) -> List[str]:
    return re.findall(r"\d+(?:[.,]\d+)?", t)


def extract_entities(t: str) -> List[str]:
    return re.findall(r"\b[A-Z][A-Za-z0-9_]+\b|\b\w+_\w+\b|\b[A-Z0-9_]{2,}\b", t)


def missing_terms(src_ko: str, tgt_en: str, ko2en: Dict[str, str]) -> List[str]:
    src_tok = ko_tokens(src_ko)
    cand = [ko2en[k] for k in src_tok if k in ko2en]
    tgt_words = {w.lower() for w in re.findall(r"\w+", tgt_en)}
    return [term for term in cand if term.lower() not in tgt_words]

# ────────────────────────────────────
# GPT wrappers
# ────────────────────────────────────
@retry(retry=retry_if_exception_type((OpenAIError, APIConnectionError, Timeout)),
       wait=wait_exponential(min=2, max=20), stop=stop_after_attempt(6))
def chat(msgs: List[Dict[str, str]], timeout: int = 60):
    return openai.ChatCompletion.create(model=MODEL_CHAT, messages=msgs, temperature=0, request_timeout=timeout)

def main() -> None:
    cache: Dict[str, Any] = load_json(CACHE_PATH) if CACHE_PATH.exists() else {}
    term_pairs = load_json(TB_JSON)
    ko2en = {p["ko"]: p["en-US"] for p in term_pairs}

    ko_dict = load_json(KO_JSON)
    en_dict = load_json(EN_JSON)
    reuse_tm = load_json(REUSE_TM_PATH)
    keys = list(ko_dict)[:100]

    results: List[Dict[str, Any]] = []
    gpt_queue: List[Tuple[str, str, str]] = []

    for k in tqdm(keys, desc="similarity+rules"):
        src_ko = ko_dict[k]
        tgt_en = en_dict.get(k, "")
        if not tgt_en:
            gpt_queue.append((k, src_ko, tgt_en)); continue

        sim = cosine(
            EMBED_MODEL.encode(src_ko, normalize_embeddings=False),
            EMBED_MODEL.encode(tgt_en, normalize_embeddings=False),
        )
        miss_nums = list(set(extract_numbers(src_ko)) - set(extract_numbers(tgt_en)))
        miss_ents = list(set(extract_entities(src_ko)) - set(extract_entities(tgt_en)))
        miss_terms = missing_terms(src_ko, tgt_en, ko2en)
        reuse_ratio = fuzzy_ratio(reuse_tm.get(k, ""), tgt_en)

        need_gpt = sim < THRESH_SIM or miss_nums or miss_ents or miss_terms
        base_info = {
            "key": k,
            "semantic_sim": round(float(sim), 3),  # cast to built‑in float
            "missing_numbers": miss_nums,
            "missing_entities": miss_ents,
            "missing_terms": miss_terms,
            "reuse_ratio": round(float(reuse_ratio), 3),  # cast to built‑in float
        }
        if need_gpt:
            gpt_queue.append((k, src_ko, tgt_en))
            results.append({**base_info, "gpt_needed": True})
        else:
            results.append({**base_info, "gpt_needed": False, "overall": 100, "adequacy": 100, "fluency": 100, "comments": [], "alternatives": []})

    with gzip.open(OUTPUT_PATH, "wt", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    print("done →", OUTPUT_PATH)

if __name__ == "__main__":
    main()
