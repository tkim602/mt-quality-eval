from __future__ import annotations

import asyncio, os, random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import numpy as np, orjson, torch
from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError, APIError
from sentence_transformers import SentenceTransformer
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

import cfg

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TERM = ", ".join(f"{k}→{v}" for k, v in cfg.TERMBASE.items())

PROMPTS: Dict[str, str] = {
    "soft_pass": (
        "You are a professional English copy‑editor specializing in cybersecurity and software documentation.\n"
        "TASK: Refine the machine‑translated (MT) sentence for clarity, conciseness, and natural fluency while 100 % preserving placeholders like {{0}}, {{name}}, etc.\n"   # ← {{ }}
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

async def _call_gpt(prompt: str, mode: str, retry: int = 5) -> str:
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
                await asyncio.sleep(2 ** attempt + random.random())
        raise RuntimeError("OpenAI call failed after retries")

async def edit_sentence(src: str, mt: str, mode: str, evidence: str) -> str:
    prompt = PROMPTS[mode].format(evidence=evidence) + f"\n\nSRC (ko): {src}\nMT  (en): {mt}"
    return await _call_gpt(prompt, mode)

_labse = SentenceTransformer(getattr(cfg, "COS_MODEL", "sentence-transformers/LaBSE")).to(getattr(cfg, "DEVICE", "cpu"))

try:
    from comet import load_from_checkpoint, download_model
    _comet = load_from_checkpoint(download_model(getattr(cfg, "COMET_CKPT", "Unbabel/wmt22-cometkiwi-da")))
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
    preds = _comet.predict(data, batch_size=16, gpus=0 if getattr(cfg, "DEVICE", "cpu") == "cpu" else 1)
    return [float(s) for s in preds["scores"]]

async def main() -> None:
    in_path = cfg.OUT_DIR / "gemba_v3.json"
    out_path = cfg.OUT_DIR / "ape_evidence_based_v3.json"

    items = orjson.loads(in_path.read_bytes())
    targets_idx = [i for i, r in enumerate(items) if r.get("tag") in ("soft_pass", "fail")]

    async def process(idx: int):
        rec = items[idx]
        ev  = rec.get("_ev", "n/a")       
        rec["ape"] = await edit_sentence(rec["src"], rec["mt"], rec["tag"], ev)


    await tqdm_asyncio.gather(*(process(i) for i in targets_idx), total=len(targets_idx), desc="APE edits")

    src_txt = [items[i]["src"] for i in targets_idx]
    ape_txt = [items[i]["ape"] for i in targets_idx]
    ape_cos = cosine_batch(src_txt, ape_txt)
    ape_comet = comet_batch(src_txt, ape_txt)

    for idx, cos_v, comet_v in zip(tqdm(targets_idx, desc="Merging metrics"), ape_cos, ape_comet):
        rec = items[idx]
        rec["ape_cos"] = float(cos_v)
        rec["ape_comet"] = float(comet_v)
        rec["delta_cos"] = rec["ape_cos"] - rec.get("cos", float("nan"))
        rec["delta_comet"] = rec["ape_comet"] - rec.get("comet", float("nan"))

    ordered = []
    for rec in items:
        od = OrderedDict()
        for k in ("key", "src", "mt", "ape", "ape_cos", "delta_cos", "ape_comet", "delta_comet"):
            if k in rec:
                od[k] = rec[k]
        for k, v in rec.items():
            if k not in od:
                od[k] = v
        ordered.append(od)

    out_path.write_bytes(orjson.dumps(ordered, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))

if __name__ == "__main__":
    asyncio.run(main())
