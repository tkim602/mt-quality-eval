from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import orjson
import torch
from dotenv import load_dotenv
from openai import AsyncOpenAI, APIError, RateLimitError
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

import cfg
from q_score import (
    apply_bucket_safety_belt,
    calculate_q_score,
    compute_bucket_percentiles,
    compute_global_stats,
    q_to_temp_grade,
    temp_grade_to_final_grade,
    z_standardize,
)

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_termbase() -> Dict[str, str]:
    if hasattr(cfg, "TERMBASE_PATH") and Path(cfg.TERMBASE_PATH).exists():
        with open(cfg.TERMBASE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return {item["ko"]: item["en-US"] for item in data if "ko" in item and "en-US" in item}
            return data
    return {}

TERMS: Dict[str, str] = load_termbase()
TERM_STR = ", ".join(f"{k}→{v}" for k, v in TERMS.items())


def get_prompt(mode: str, evidence: str) -> str:
    common = (
        "Omit unnecessary articles (a/an/the) and pronouns.\n"
        "Do not add, remove, reorder CVE IDs, paths, numbers, or option names.\n"
        "Replace a term only if it is demonstrably wrong and the Korean source proves it.\n"
        "Start with the given machine translation (mt) as the **baseline**.\n"
        "Make only the edits needed to fix errors in meaning, terminology, grammar, punctuation, or style.\n"
    )
    if mode == "soft_pass":
        return (
            "You are a professional English copy‑editor specializing in cybersecurity and software documentation.\n"
            "TASK: Refine the machine‑translated (MT) sentence for clarity, conciseness, and natural fluency while 100 % preserving placeholders like {{0}}, {{name}}, etc.\n"
            "Keep technical terminology intact. Do NOT change the sentence meaning.\n"
            f"Term base: {TERM_STR}\n"
            f"EVIDENCE of main issue(s): {evidence}\n"
            + common +
            "Return ONLY the improved English sentence."
        )
    return (
        "You are a senior technical translator specializing in cybersecurity and software documentation.\n"
        "TASK: Post‑edit the machine‑translated (MT) sentence so that it accurately and clearly conveys the source Korean meaning in natural, professional English.\n"
        f"Term base: {TERM_STR}\n"
        f"EVIDENCE of main issue(s): {evidence}\n"
        + common +
        "If the MT sentence sounds ridiculously wrong, you may reorder the structure and rewrite the sentence, **but** you must preserve all important information.\n"
        "Placeholders such as {{0}}, {{name}}, {{path}} must appear exactly as in the MT.\n"
        "Return ONLY the fully improved English sentence."
    )

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
            except (RateLimitError, APIError):
                wait = min(60, 2 ** attempt + random.uniform(1, 3))
                logger.warning("Rate‑limited (attempt %d/%d). Sleeping %.1fs…", attempt + 1, retry, wait)
                await asyncio.sleep(wait)
        logger.error("OpenAI call failed after %d retries – returning placeholder.", retry)
        return "[APE_FAILED_RATE_LIMIT]"

async def edit_sentence(src: str, mt: str, mode: str, evidence: str) -> str:
    prompt = get_prompt(mode, evidence) + f"\n\nSRC (ko): {src}\nMT  (en): {mt}"
    return await _call_gpt(prompt, mode)

DEVICE = getattr(cfg, "DEVICE", "cpu")
_koe5 = SentenceTransformer(getattr(cfg, "COS_MODEL", "nlpai-lab/KoE5")).to(DEVICE)

try:
    from comet import download_model, load_from_checkpoint
    _comet = load_from_checkpoint(download_model(getattr(cfg, "COMET_CKPT", "Unbabel/wmt22-cometkiwi-da"))).to(DEVICE)
    _comet.eval()
    _has_comet = True
except Exception:
    logger.warning("COMET model could not be loaded – comet scores will be NaN.")
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
    preds = _comet.predict(data, batch_size=16, gpus=0, progress_bar=False)
    return [float(s) for s in preds["scores"]]

async def main() -> None:
    run_dir_env = os.getenv("RUN_DIR")
    if run_dir_env:
        run_path = Path(run_dir_env)
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
        logger.error("Input file not found: %s", in_path)
        return

    items: List[dict] = orjson.loads(in_path.read_bytes())

    targets_idx = [i for i, r in enumerate(items) if r.get("tag") in ("soft_pass", "fail")]

    async def process(idx: int):
        record = items[idx]
        evidence = record.get("flag", {}).get("gemba_reason", "n/a")
        record["ape"] = await edit_sentence(record["src"], record["mt"], record["tag"], evidence)

    await tqdm_asyncio.gather(*(process(i) for i in targets_idx), total=len(targets_idx), desc="APE edits")

    ape_indices = [i for i in targets_idx if "ape" in items[i]]
    src_txt = [items[i]["src"] for i in ape_indices]
    ape_txt = [items[i]["ape"] for i in ape_indices]

    ape_cos = cosine_batch(src_txt, ape_txt)
    ape_com = comet_batch(src_txt, ape_txt)

    for i, c, m in zip(ape_indices, ape_cos, ape_com):
        original = items[i]
        original["ape_cos"] = float(c)
        original["delta_cos"] = float(c - original.get("cos", 0.0))
        original["ape_comet"] = float(m)
        original["delta_comet"] = float(m - original.get("comet", 0.0))

    from gemba_batch import gemba_batch

    logger.info("Computing GEMBA for APE sentences…")
    ape_gemba_results = await gemba_batch(src_txt, ape_txt)

    for i, result in zip(ape_indices, ape_gemba_results):
        original_gemba = items[i].get("gemba", 0.0)
        ape_gemba = float(result[0] if isinstance(result, tuple) else result.get("overall", 0.0))
        record = items[i]
        record["ape_gemba"] = ape_gemba
        record["delta_gemba"] = float(ape_gemba - original_gemba)

    if getattr(cfg, "USE_Q_SCORE", False) and ape_indices:
        logger.info("Computing Q‑scores for APE sentences…")

        all_for_stats = []
        for rec in items:
            all_for_stats.append({"cos": rec.get("cos", 0), "comet": rec.get("comet", 0), "gemba": rec.get("gemba", 0)})
            if "ape_cos" in rec:
                all_for_stats.append({"cos": rec["ape_cos"], "comet": rec["ape_comet"], "gemba": rec["ape_gemba"]})

        global_stats = compute_global_stats(all_for_stats)
        logger.debug("Global stats: %s", global_stats)

        bucket_records = [
            {"bucket": rec.get("bucket", "medium"), "cos": rec.get("ape_cos"), "comet": rec.get("ape_comet")}
            for rec in items if "ape_cos" in rec
        ]
        bucket_q20 = compute_bucket_percentiles(bucket_records)

        for idx in ape_indices:
            rec = items[idx]
            bucket = rec.get("bucket", "medium")
            z_cos = z_standardize(rec["ape_cos"], global_stats["cos"]["mean"], global_stats["cos"]["std"])
            z_comet = z_standardize(rec["ape_comet"], global_stats["comet"]["mean"], global_stats["comet"]["std"])
            z_gemba = z_standardize(rec["ape_gemba"], global_stats["gemba"]["mean"], global_stats["gemba"]["std"])

            q_score = calculate_q_score(z_cos, z_comet, z_gemba)
            temp_grade = q_to_temp_grade(q_score)
            final_temp = apply_bucket_safety_belt(temp_grade, rec["ape_cos"], rec["ape_comet"], rec["ape_gemba"], bucket, bucket_q20)
            final_grade = temp_grade_to_final_grade(final_temp)

            rec.update(
                {
                    "ape_z_cos": float(z_cos),
                    "ape_z_comet": float(z_comet),
                    "ape_z_gemba": float(z_gemba),
                    "ape_q_score": float(q_score),
                    "ape_temp_grade": temp_grade,
                    "ape_final_temp_grade": final_temp,
                    "ape_tag": final_grade,
                    "ape_q_score_info": {
                        "global_stats": {
                            "cos": {"mean": float(global_stats["cos"]["mean"]), "std": float(global_stats["cos"]["std"])},
                            "comet": {"mean": float(global_stats["comet"]["mean"]), "std": float(global_stats["comet"]["std"])},
                            "gemba": {"mean": float(global_stats["gemba"]["mean"]), "std": float(global_stats["gemba"]["std"])}
                        },
                        "bucket_q20": {k: float(v) for k, v in bucket_q20.get(bucket, {}).items()},
                        "downgraded": temp_grade != final_temp,
                    },
                }
            )

    final = []
    for r in items:
        od = OrderedDict()
        for k in (
            "key","src","mt","ape","validation","ape_cos","delta_cos","ape_comet","delta_comet",
            "ape_gemba","delta_gemba","ape_q_score","ape_z_cos","ape_z_comet","ape_z_gemba",
            "ape_temp_grade","ape_final_temp_grade","ape_tag","ape_q_score_info"
        ):
            if k in r:
                od[k] = r[k]
        for k,v in r.items():
            if k not in od:
                od[k] = v
        final.append(od)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(orjson.dumps(final, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    logger.info("Done → %s", out_path)


if __name__ == "__main__":
    asyncio.run(main())
