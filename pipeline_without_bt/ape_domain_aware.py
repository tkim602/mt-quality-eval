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

def detect_domain_from_filename(filename: str) -> str:
    """
    Detect domain from input filename pattern.
    Examples:
    - en_checker.json -> sparrow
    - en_conversation.json -> conversation 
    - en_academic.json -> academic
    - en_news.json -> news_article
    - en_fiction.json -> fiction
    - en_poetry.json -> poetry
    - en_sns.json -> sns
    """
    filename = filename.lower()
    
    # Define domain mapping from filename patterns
    domain_patterns = {
        'checker': 'sparrow',
        'sparrow': 'sparrow',
        'conversation': 'conversation',
        'academic': 'academic', 
        'news': 'news_article',
        'fiction': 'fiction',
        'poetry': 'poetry',
        'sns': 'sns'
    }
    
    # Check each pattern in the filename
    for pattern, domain in domain_patterns.items():
        if pattern in filename:
            return domain
    
    # Default to sparrow if no pattern matches
    logger.warning(f"No domain pattern found in filename '{filename}', defaulting to 'sparrow'")
    return 'sparrow'

def load_domain_prompts(domain: str) -> Dict[str, str]:
    """
    Load prompts for the specified domain from the prompts folder.
    """
    prompts_dir = Path(__file__).parent / "prompts" / domain
    
    if not prompts_dir.exists():
        logger.warning(f"Domain prompts directory not found: {prompts_dir}, using sparrow as fallback")
        prompts_dir = Path(__file__).parent / "prompts" / "sparrow"
    
    prompts = {}
    
    # Load soft_pass prompt
    soft_pass_file = prompts_dir / f"{domain}_soft_pass.txt"
    if soft_pass_file.exists():
        with open(soft_pass_file, 'r', encoding='utf-8') as f:
            prompts["soft_pass"] = f.read().strip()
    
    # Load fail prompt  
    fail_file = prompts_dir / f"{domain}_fail.txt"
    if fail_file.exists():
        with open(fail_file, 'r', encoding='utf-8') as f:
            prompts["fail"] = f.read().strip()
    
    # Fallback to sparrow if prompts not found
    if not prompts:
        logger.warning(f"No prompts found for domain '{domain}', using sparrow fallback")
        sparrow_dir = Path(__file__).parent / "prompts" / "sparrow"
        
        soft_pass_file = sparrow_dir / "sparrow_soft_pass.txt"
        if soft_pass_file.exists():
            with open(soft_pass_file, 'r', encoding='utf-8') as f:
                prompts["soft_pass"] = f.read().strip()
        
        fail_file = sparrow_dir / "sparrow_fail.txt"
        if fail_file.exists():
            with open(fail_file, 'r', encoding='utf-8') as f:
                prompts["fail"] = f.read().strip()
    
    return prompts

def get_prompt(mode: str, evidence: str, domain_prompts: Dict[str, str]) -> str:
    """
    Get the appropriate prompt for the given mode, with evidence and term base substituted.
    """
    template = domain_prompts.get(mode, domain_prompts.get("fail", ""))
    
    if not template:
        logger.error(f"No prompt template found for mode '{mode}'")
        return ""
    
    # Substitute placeholders
    prompt = template.replace("{TERM}", TERM).replace("{EVIDENCE}", evidence)
    return prompt

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

async def edit_sentence(src: str, mt: str, mode: str, evidence: str, domain_prompts: Dict[str, str]) -> str:
    prompt = get_prompt(mode, evidence, domain_prompts) + f"\\n\\nSRC (ko): {src}\\nMT  (en): {mt}"
    return await _call_gpt(prompt, mode)

_koe5 = SentenceTransformer(getattr(cfg, "COS_MODEL", "nlpai-lab/KoE5")).to(getattr(cfg, "DEVICE", "cpu"))

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
    a = _koe5.encode(src, convert_to_tensor=True, normalize_embeddings=True, batch_size=32)
    b = _koe5.encode(tgt, convert_to_tensor=True, normalize_embeddings=True, batch_size=32)
    return (a * b).sum(-1).cpu().numpy()

def comet_batch(src: List[str], tgt: List[str]) -> List[float]:
    if not _has_comet:
        return [float("nan")] * len(src)
    data = [{"src": s, "mt": t} for s, t in zip(src, tgt)]
    preds = _comet.predict(data, batch_size=16, gpus=0, progress_bar=True)
    return [float(s) for s in preds["scores"]]

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
    
    # Detect domain from data or environment variable, fallback to filename detection
    domain = None
    if items and len(items) > 0:
        # Use domain from first record if available
        domain = items[0].get("domain")
    
    if not domain:
        # Try environment variable set by filter.py
        domain = os.getenv('PIPELINE_DOMAIN')
    
    if not domain:
        # Fallback to filename detection
        domain = detect_domain_from_filename(input_filename)
        logger.warning(f"No domain found in data or environment, detected from filename: {domain}")
    else:
        logger.info(f"Using domain from pipeline: {domain}")
    
    # Load domain-specific prompts
    domain_prompts = load_domain_prompts(domain)
    logger.info(f"Loaded prompts for domain: {domain}")
    logger.info(f"Available prompt modes: {list(domain_prompts.keys())}")
    targets_idx = [i for i, r in enumerate(items) if r.get("tag") in ("soft_pass", "fail")]

    async def process(i: int):
        r = items[i]
        ev = r.get("flag", {}).get("gemba_reason", "n/a")
        r["ape"] = await edit_sentence(r["src"], r["mt"], r["tag"], ev, domain_prompts)

    await tqdm_asyncio.gather(
        *(process(i) for i in targets_idx), total=len(targets_idx), desc=f"APE edits ({domain})"
    )

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

    from gemba_batch import gemba_batch
    logger.info(f"APE 개선 후 GEMBA 점수 계산 중... (domain: {domain})")
    
    ape_gemba_results = await gemba_batch(src_txt, ape_txt)
    for i, gemba_result in zip(ape_indices, ape_gemba_results):
        original_gemba = items[i].get("gemba", 0.0)
        if isinstance(gemba_result, tuple):
            ape_gemba = float(gemba_result[0]) 
        else:
            ape_gemba = gemba_result.get("overall", 0.0)
        items[i]["ape_gemba"] = float(ape_gemba)
        items[i]["delta_gemba"] = float(ape_gemba - original_gemba)

    final = []
    for r in items:
        od = OrderedDict()
        # Add domain info to output
        od["domain"] = domain
        for k in (
            "key","src","mt","ape","validation","ape_cos","delta_cos","ape_comet","delta_comet",
            "ape_gemba","delta_gemba"
        ):
            if k in r:
                od[k] = r[k]
        for k,v in r.items():
            if k not in od:
                od[k] = v
        final.append(od)

    out_path.write_bytes(orjson.dumps(final, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    logger.info(f"Done → {out_path} (domain: {domain})")

if __name__ == "__main__":
    asyncio.run(main())
