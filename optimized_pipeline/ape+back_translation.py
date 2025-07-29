# ape_plus_back_translation.py
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np
import orjson
import torch
from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError, APIError
from sentence_transformers import SentenceTransformer
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

import cfg

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_termbase():
    if hasattr(cfg, 'TERMBASE_PATH') and Path(cfg.TERMBASE_PATH).exists():
        with open(cfg.TERMBASE_PATH, 'r', encoding='utf-8') as f:
            termbase_data = json.load(f)
            if isinstance(termbase_data, list):
                return {item['ko']: item['en-US'] for item in termbase_data if 'ko' in item and 'en-US' in item}
            else:
                return termbase_data
    return {}

termbase = load_termbase()
TERM = ", ".join(f"{k}→{v}" for k, v in termbase.items())

def get_prompt(mode: str, evidence: str) -> str:
    prompts = {
        "soft_pass": (
            "You are a professional English copy‑editor specializing in cybersecurity and software documentation.\n"
            "TASK: Refine the machine‑translated (MT) sentence for clarity, conciseness, and natural fluency while 100 % preserving placeholders like {{0}}, {{name}}, etc.\n"
            "Keep technical terminology intact. Do NOT change the sentence meaning.\n"
            f"Term base: {TERM}\n"
            f"EVIDENCE of main issue(s): {evidence}\n"
            "Omit unnecessary articles (a/an/the) and pronouns.\n"
            "Do not add, remove, reorder CVE IDs, paths, numbers, or option names.\n"
            "Replace a term only if it is demonstrably wrong and the Korean source proves it.\n"
            "Start with the given machine translation (mt) as the **baseline**.\n"
            "Make only the edits needed to fix errors in meaning, terminology, grammar, punctuation, or style.\n"
            "Return ONLY the improved English sentence."
        ),
        "fail": (
            "You are a senior technical translator specializing in cybersecurity and software documentation.\n"
            "TASK: Post‑edit the machine‑translated (MT) sentence so that it accurately and clearly conveys the source Korean meaning in natural, professional English.\n"
            f"Term base: {TERM}\n"
            f"EVIDENCE of main issue(s): {evidence}\n"
            "Omit unnecessary articles (a/an/the) and pronouns.\n"
            "Do not add, remove, reorder CVE IDs, paths, numbers, or option names.\n"
            "Replace a term only if it is demonstrably wrong and the Korean source proves it.\n"
            "Start with the given machine translation (mt) as the **baseline**.\n"
            "Make only the edits needed to fix errors in meaning, terminology, grammar, punctuation, or style.\n"
            "If the MT sentence sounds ridiculously wrong, you may reorder the structure and rewrite the sentence, **but** you must preserve all important information.\n"
            "Placeholders such as {{0}}, {{name}}, {{path}} must appear exactly as in the MT.\n"
            "Return ONLY the fully improved English sentence."
        ),
    }
    return prompts.get(mode, prompts["fail"])

PROMPTS: Dict[str, str] = {
    "soft_pass": (
        "You are a professional English copy‑editor specializing in cybersecurity and software documentation.\n"
        "TASK: Refine the machine‑translated (MT) sentence for clarity, conciseness, and natural fluency while 100 % preserving placeholders like {{0}}, {{name}}, etc.\n"
        "Keep technical terminology intact. Do NOT change the sentence meaning.\n"
        f"Term base: {TERM}\n"
        "EVIDENCE of main issue(s): {evidence}\n"
        "Omit unnecessary articles (a/an/the) and pronouns.\n"
        "Do not add, remove, reorder CVE IDs, paths, numbers, or option names.\n"
        "Replace a term only if it is demonstrably wrong and the Korean source proves it.\n"
        "Start with the given machine translation (mt) as the **baseline**.\n"
        "Make only the edits needed to fix errors in meaning, terminology, grammar, punctuation, or style.\n"
        "Return ONLY the improved English sentence."
    ),
    "fail": (
        "You are a senior technical translator specializing in cybersecurity and software documentation.\n"
        "TASK: Post‑edit the machine‑translated (MT) sentence so that it accurately and clearly conveys the source Korean meaning in natural, professional English.\n"
        f"Term base: {TERM}\n"
        "EVIDENCE of main issue(s): {evidence}\n"
        "Omit unnecessary articles (a/an/the) and pronouns.\n"
        "Do not add, remove, reorder CVE IDs, paths, numbers, or option names.\n"
        "Replace a term only if it is demonstrably wrong and the Korean source proves it.\n"
        "Start with the given machine translation (mt) as the **baseline**.\n"
        "Make only the edits needed to fix errors in meaning, terminology, grammar, punctuation, or style.\n"
        "If the MT sentence sounds ridiculously wrong, you may reorder the structure and rewrite the sentence, **but** you must preserve all important information.\n"
        "Placeholders such as {{0}}, {{name}}, {{path}} must appear exactly as in the MT.\n"
        "Return ONLY the fully improved English sentence."
    ),
}

SEM = asyncio.Semaphore(getattr(cfg, "APE_CONCURRENCY", 8))

async def _call_gpt(prompt: str, mode: str, retry: int = 8) -> str:  
    async with SEM:
        for attempt in range(retry):
            try:
                resp = await client.chat.completions.create(
                    model=cfg.APE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1 if mode == "soft_pass" else 0.3,
                )
                return resp.choices[0].message.content.strip()
            except (RateLimitError, APIError) as e:
                wait_time = min(60, 2 ** attempt + random.uniform(1, 3))  
                logger.warning(f"Rate limit hit (attempt {attempt+1}/{retry}). Waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
        
        logger.error(f"OpenAI call failed after {retry} retries. Returning placeholder.")
        return "[APE_FAILED_RATE_LIMIT]"

async def edit_sentence(src: str, mt: str, mode: str, evidence: str) -> str:
    prompt = get_prompt(mode, evidence) + f"\n\nSRC (ko): {src}\nMT  (en): {mt}"
    return await _call_gpt(prompt, mode)

_labse = SentenceTransformer(getattr(cfg, "COS_MODEL", "sentence-transformers/LaBSE")).to(getattr(cfg, "DEVICE", "cpu"))

try:
    from comet import load_from_checkpoint, download_model
    _comet = load_from_checkpoint(download_model(getattr(cfg, "COMET_CKPT", "Unbabel/wmt22-cometkiwi-da"))).to(getattr(cfg, "DEVICE", "cpu"))
    _comet.eval()
    _has_comet = True
except Exception:
    _has_comet = False

@torch.no_grad()
def cosine_batch(src: List[str], tgt: List[str]) -> np.ndarray:
    if not src:
        return np.empty(0, dtype=np.float32)
    a = _labse.encode(src, convert_to_tensor=True, normalize_embeddings=True, batch_size=32)
    b = _labse.encode(tgt, convert_to_tensor=True, normalize_embeddings=True, batch_size=32)
    return (a * b).sum(-1).cpu().numpy()

def comet_batch(src: List[str], tgt: List[str]) -> List[float]:
    if not _has_comet:
        return [float("nan")] * len(src)
    data = [{"src": s, "mt": t} for s, t in zip(src, tgt)]
    preds = _comet.predict(data, batch_size=16, gpus=0, progress_bar=True)
    return [float(s) for s in preds["scores"]]

def _bt_prompt(n: int) -> str:
    return (
        "You are a professional EN→KO translator specialising in cybersecurity "
        "and software‑engineering documentation.\n\n"
        f"Back‑translate each English sentence below into Korean, preserving placeholders exactly.\n"
        f"Return numbered lines like `1. …`, `2. …` (length {n})."
    )

_LINE_SPLIT = re.compile(r"[,:|\t\-]|")

async def _bt_call(batch: List[str]) -> List[str]:
    msgs = [
        {"role": "system", "content": _bt_prompt(len(batch))},
        {"role": "user",   "content": "\n".join(f"{i+1}. {s}" for i, s in enumerate(batch))},
    ]
    for attempt in range(5):
        try:
            async with SEM:
                resp = await client.chat.completions.create(
                    model=cfg.BT_MODEL if hasattr(cfg, "BT_MODEL") else "gpt-4o-mini",
                    messages=msgs,
                    temperature=0.0,
                    max_tokens=1024,
                )
            text = resp.choices[0].message.content
            print("BT raw response:", text)  
            outs = [""] * len(batch)
            for ln in text.splitlines():
                if not ln.strip():
                    continue
                idx_str, sep, rest = ln.partition(".")
                if not sep or not idx_str.isdigit():
                    continue
                i = int(idx_str) - 1
                if 0 <= i < len(batch):
                    outs[i] = rest.strip()
            if any(outs):
                final_outs = [o if o else "BT_PARSE_FAILED" for o in outs]
                return final_outs
            raise ValueError("BT parse fail - no lines parsed")
        except (RateLimitError, APIError, ValueError) as e:
            logger.warning(f"BT call failed on attempt {attempt+1}: {e}")
            await asyncio.sleep(2 ** attempt + random.random())
    
    logger.error("BT failed after retries, returning placeholders.")
    return ["BT_FAILED_AFTER_RETRIES"] * len(batch)

async def main():
    run_dir = os.getenv('RUN_DIR')
    if run_dir:
        run_path = Path(run_dir)
        input_filename = cfg.GEMBA_OUTPUT_FILENAME
        output_filename = cfg.APE_OUTPUT_FILENAME
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_path = cfg.OUT_DIR
        input_filename = f"gemba_{timestamp}.json"
        output_filename = f"ape_evidence_{timestamp}.json"
    
    in_path = run_path / input_filename
    out_path = run_path / output_filename

    if not in_path.exists():
        logger.error(f"Input file not found: {in_path}")
        return

    items = orjson.loads(in_path.read_bytes())
    targets_idx = [i for i, r in enumerate(items) if r.get("tag") in ("soft_pass", "fail")]

    async def process(i: int):
        r = items[i]
        ev = r.get("flag", {}).get("gemba_reason", "n/a")
        r["ape"] = await edit_sentence(r["src"], r["mt"], r["tag"], ev)

    await tqdm_asyncio.gather(
        *(process(i) for i in targets_idx), total=len(targets_idx), desc="APE edits"
    )

    # APE가 있는 항목들만 필터링해서 cosine, comet 계산
    ape_indices = [i for i in targets_idx if "ape" in items[i]]
    src_txt = [items[i]["src"] for i in ape_indices]
    ape_txt = [items[i]["ape"] for i in ape_indices]
    ape_cos = cosine_batch(src_txt, ape_txt)
    ape_com = comet_batch(src_txt, ape_txt)
    for i, c, m in zip(ape_indices, ape_cos, ape_com):
        items[i]["ape_cos"]   = float(c)
        items[i]["ape_comet"] = float(m)
        items[i]["delta_cos"] = float(c - items[i].get("cos", 0.0))
        items[i]["delta_comet"] = float(m - items[i].get("comet", 0.0))

    # APE 개선 후 GEMBA 점수 계산
    from gemba_batch import gemba_batch
    logger.info("APE 개선 후 GEMBA 점수 계산 중...")
    
    # APE된 텍스트에 대해 GEMBA 평가 수행 (이미 위에서 정의된 src_txt, ape_txt 사용)
    ape_gemba_results = await gemba_batch(src_txt, ape_txt)
    for i, gemba_result in zip(ape_indices, ape_gemba_results):
        original_gemba = items[i].get("gemba", 0.0)
        # gemba_batch returns tuples (overall, adequacy, fluency, evidence)
        if isinstance(gemba_result, tuple):
            ape_gemba = float(gemba_result[0])  # overall score is first element
        else:
            ape_gemba = gemba_result.get("overall", 0.0)
        items[i]["ape_gemba"] = float(ape_gemba)
        items[i]["delta_gemba"] = float(ape_gemba - original_gemba)

    batches = [ape_indices[i : i + cfg.GEMBA_BATCH] for i in range(0, len(ape_indices), cfg.GEMBA_BATCH)]
    for batch in tqdm(batches, desc="Back‑translate"):
        ape_en = [items[i]["ape"] for i in batch]
        mt_en  = [items[i]["mt"]  for i in batch]
        ape_bt = await _bt_call(ape_en)
        mt_bt  = await _bt_call(mt_en)
        for idx, bt1, bt2 in zip(batch, ape_bt, mt_bt):
            items[idx]["ape_bt"] = bt1
            items[idx]["mt_bt"]  = bt2

    src_ko      = [items[i]["src"]    for i in ape_indices]
    ape_bt_list = [items[i]["ape_bt"] for i in ape_indices]
    mt_bt_list  = [items[i]["mt_bt"]  for i in ape_indices]

    ape_bt_cos = cosine_batch(src_ko, ape_bt_list)
    mt_bt_cos  = cosine_batch(src_ko, mt_bt_list)
    ape_bt_com = comet_batch(src_ko, ape_bt_list)
    mt_bt_com  = comet_batch(src_ko, mt_bt_list)

    for i, c1, c2, d1, d2 in zip(ape_indices, ape_bt_cos, mt_bt_cos, ape_bt_com, mt_bt_com):
        r = items[i]
        r["ape_bt_cos"]    = float(c1)
        r["mt_bt_cos"]     = float(c2)
        r["delta_bt_cos"]  = float(c1 - c2)
        r["ape_bt_comet"]    = float(d1)
        r["mt_bt_comet"]     = float(d2)
        r["delta_bt_comet"] = float(d1 - d2)

    final = []
    for r in items:
        od = OrderedDict()
        for k in (
            "key","src","mt","ape","validation","ape_cos","delta_cos","ape_comet","delta_comet",
            "ape_gemba","delta_gemba","ape_bt_cos","mt_bt_cos","delta_bt_cos","ape_bt_comet","mt_bt_comet","delta_bt_comet"
        ):
            if k in r:
                od[k] = r[k]
        for k,v in r.items():
            if k not in od:
                od[k] = v
        final.append(od)

    out_path.write_bytes(orjson.dumps(final, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    logger.info("Done → %s", out_path)

if __name__ == "__main__":
    asyncio.run(main())
