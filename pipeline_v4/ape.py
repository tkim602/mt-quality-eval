"""ape.py — COMET 로드 실패 대응 + 선택적 계산

변경 사항
────────────────────────────────────────────────────────────────────────────
1. **COMET 체크포인트 로드 예외 처리**
   • `cfg.COMET_CKPT` 로드 실패 시 경고 후 COMET 계산 스킵.
   • `ape_comet`, `delta_comet` 필드는 스킵될 수 있음.
2. 처음 gemba.json에 이미 존재하는 `cos`, `comet` 그대로 사용.
3. 경고 메시지에 사용 가능한 기본 HF 모델 id (`Unbabel/wmt22-comet-da`) 예시.
"""
from __future__ import annotations

import asyncio
import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import orjson
from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError, APIError
from sentence_transformers import SentenceTransformer
import torch
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

import cfg

# ─── OpenAI init ───────────────────────────────────────────────────────────
load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─── Prompt templates ──────────────────────────────────────────────────────
PROMPTS: Dict[str, str] = {
    "soft_pass": (
        "You are a professional English copy‑editor specializing in cybersecurity "
        "and software documentation.\n"
        "TASK: Refine the machine‑translated (MT) sentence for clarity, conciseness, "
        "and natural fluency while 100 % preserving placeholders like {0}, {name}, etc.\n"
        "Keep technical terminology intact. Do NOT change the sentence meaning.\n"
        "Return ONLY the improved English sentence."
    ),
    "fail": (
        "You are a senior technical translator specializing in cybersecurity and software documentation.\n"
        "TASK: Rewrite the machine‑translated (MT) sentence so that it accurately and "
        "clearly conveys the source Korean meaning in natural, professional English.\n"
        "You may reorganize wording and structure but must preserve every piece of information.\n"
        "Placeholders such as {0}, {name}, {path} must appear exactly as in the MT.\n"
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

async def edit_sentence(src: str, mt: str, mode: str) -> str:
    prompt = (
        PROMPTS[mode]
        + "\n\n"
        + f"SRC (ko): {src}\n"
        + f"MT  (en): {mt}"
    )
    return await _call_gpt(prompt, mode)

_labse = SentenceTransformer(getattr(cfg, "COS_MODEL", "sentence-transformers/LaBSE")).to(
    getattr(cfg, "DEVICE", "cpu")
)

try:
    from comet import load_from_checkpoint

    _comet = load_from_checkpoint(getattr(cfg, "COMET_CKPT", "Unbabel/wmt22-cometkiwi-da"))
    _comet.eval()
    _has_comet = True
except Exception as e: 
    print(
        f"[WARN] COMET checkpoint load failed: {e}\n"
        "       COMET metrics will be skipped. Set cfg.COMET_CKPT to a valid HF id, "
        "e.g. 'Unbabel/wmt22-comet-da'."
    )
    _has_comet = False

@torch.no_grad()
def cosine_batch(src: List[str], tgt: List[str]) -> np.ndarray:
    a = _labse.encode(src, convert_to_tensor=True, normalize_embeddings=True, batch_size=32)
    b = _labse.encode(tgt, convert_to_tensor=True, normalize_embeddings=True, batch_size=32)
    return (a * b).sum(1).cpu().numpy()


def comet_batch(src: List[str], tgt: List[str]) -> List[float]:
    if not _has_comet:
        return [float("nan")] * len(src)
    data = [{"src": s, "mt": t} for s, t in zip(src, tgt)]
    preds = _comet.predict(data, batch_size=16, gpus=0 if getattr(cfg, "DEVICE", "cpu") == "cpu" else 1)
    return [float(s) for s in preds["scores"]]

async def main() -> None:
    in_path: Path = cfg.OUT_DIR / "gemba.json"
    out_path: Path = cfg.OUT_DIR / "ape.json"

    items: List[dict] = orjson.loads(in_path.read_bytes())
    targets_idx = [i for i, r in enumerate(items) if r.get("result") in ("soft_pass", "fail")]

    print(f"🔹 Post‑edit targets: {len(targets_idx)} (soft_pass/fail)")

    async def process(idx: int):
        rec = items[idx]
        rec["ape"] = await edit_sentence(rec["src"], rec["mt"], rec["result"])

    await tqdm_asyncio.gather(*(process(i) for i in targets_idx), total=len(targets_idx), desc="APE edits")

    print("🔹 Computing cosine similarity …")
    src_txt = [items[i]["src"] for i in targets_idx]
    ape_txt = [items[i]["ape"] for i in targets_idx]
    ape_cos = cosine_batch(src_txt, ape_txt)

    if _has_comet:
        print("🔹 Computing COMET scores …")
        ape_comet = comet_batch(src_txt, ape_txt)
    else:
        ape_comet = [float("nan")] * len(ape_cos)

    for idx, cos_v, comet_v in zip(tqdm(targets_idx, desc="Merging metrics"), ape_cos, ape_comet):
        rec = items[idx]
        rec["ape_cos"] = round(float(cos_v), 4)
        if _has_comet:
            rec["ape_comet"] = round(float(comet_v), 4)
        rec["delta_cos"] = round(rec["ape_cos"] - rec["cos"], 4)
        if _has_comet:
            rec["delta_comet"] = round(rec["ape_comet"] - rec["comet"], 4)

    ordered: List[dict] = []
    for rec in items:
        od = OrderedDict()
        for k in ("key", "src", "mt", "ape", "ape_cos", "ape_comet", "delta_cos", "delta_comet"):
            if k in rec:
                od[k] = rec[k]
        for k, v in rec.items():
            if k not in od:
                od[k] = v
        ordered.append(od)

    out_path.write_bytes(orjson.dumps(ordered, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
