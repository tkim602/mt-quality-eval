# ape_tower2b_cpu.py  ── Automatic Post‑Editing with Tower‑Plus‑2B (CPU)

from __future__ import annotations
import asyncio, os
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import numpy as np, orjson, torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

import cfg

load_dotenv()
MODEL_NAME = "Unbabel/Tower-Plus-2B"
MAX_TOKENS = getattr(cfg, "APE_MAX_TOKENS", 96)
WORKERS    = 1
torch.set_num_threads(1)

print(f"loading {MODEL_NAME} on CPU …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    BF16 = True
except Exception as e:
    print(f"[WARN] BF16 load failed → FP32 ({e})")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    BF16 = False
model.to("cpu")

gen_cfg = GenerationConfig(
    max_new_tokens=MAX_TOKENS,
    do_sample=False,
    temperature=0.0,
    top_p=1.0,
    eos_token_id=tokenizer.eos_token_id,
)

PROMPTS: Dict[str, str] = {
    "soft_pass": (
        "You are a professional English copy‑editor specializing in cybersecurity and software documentation.\n"
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

_EXEC = ThreadPoolExecutor(max_workers=WORKERS)

def _generate(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    with torch.cpu.amp.autocast(dtype=torch.bfloat16) if BF16 else torch.no_grad():
        out_ids = model.generate(
            torch.as_tensor([input_ids]),
            generation_config=gen_cfg,
        )[0]
    return tokenizer.decode(out_ids[len(input_ids):], skip_special_tokens=True).strip()

async def edit_sentence(src: str, mt: str, mode: str) -> str:
    prompt = PROMPTS[mode] + f"\n\nSRC (ko): {src}\nMT  (en): {mt}"
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_EXEC, _generate, prompt)

_labse = SentenceTransformer(
    getattr(cfg, "COS_MODEL", "sentence-transformers/LaBSE")
).to("cpu")

try:
    from comet import download_model, load_from_checkpoint
    _comet = load_from_checkpoint(
        download_model(getattr(cfg, "COMET_CKPT", "Unbabel/wmt22-cometkiwi-da"))
    )
    _comet.eval()
    _has_comet = True
except Exception:
    _has_comet = False

@torch.no_grad()
def cosine_batch(src: List[str], tgt: List[str]) -> np.ndarray:
    if not src:
        return np.empty(0, dtype=np.float32)
    a = _labse.encode(src, convert_to_tensor=True, normalize_embeddings=True)
    b = _labse.encode(tgt, convert_to_tensor=True, normalize_embeddings=True)
    return (a * b).sum(-1).cpu().numpy()

def comet_batch(src: List[str], tgt: List[str]) -> List[float]:
    if not _has_comet:
        return [float("nan")] * len(src)
    data = [{"src": s, "mt": t} for s, t in zip(src, tgt)]
    preds = _comet.predict(data, batch_size=8, gpus=0)
    return [float(s) for s in preds["scores"]]

async def main() -> None:
    in_path  = cfg.OUT_DIR / "test_tower+2B.json"
    out_path = cfg.OUT_DIR / "ape_tower.json"

    items = orjson.loads(in_path.read_bytes())
    targets_idx = [i for i, r in enumerate(items)
                   if r.get("result") in ("soft_pass", "fail")]

    async def process(idx: int):
        rec = items[idx]
        rec["ape"] = await edit_sentence(rec["src"], rec["mt"], rec["result"])

    await tqdm_asyncio.gather(*(process(i) for i in targets_idx),
                              total=len(targets_idx), desc="APE edits")

    src_txt  = [items[i]["src"] for i in targets_idx]
    ape_txt  = [items[i]["ape"] for i in targets_idx]
    ape_cos  = cosine_batch(src_txt, ape_txt)
    ape_comet = comet_batch(src_txt, ape_txt)

    for idx, cos_v, comet_v in zip(tqdm(targets_idx, desc="Merging metrics"),
                                   ape_cos, ape_comet):
        rec = items[idx]
        rec["ape_cos"]     = float(cos_v)
        rec["ape_comet"]   = float(comet_v)
        rec["delta_cos"]   = rec["ape_cos"]   - rec.get("cos", float("nan"))
        rec["delta_comet"] = rec["ape_comet"] - rec.get("comet", float("nan"))

    ordered: List[dict] = []
    for rec in items:
        od = OrderedDict()
        for k in ("key", "src", "mt", "ape",
                  "ape_cos", "delta_cos", "ape_comet", "delta_comet"):
            if k in rec:
                od[k] = rec[k]
        for k, v in rec.items():
            if k not in od:
                od[k] = v
        ordered.append(od)

    out_path.write_bytes(orjson.dumps(
        ordered, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    print(f"[INFO] saved → {out_path}")

if __name__ == "__main__":
    asyncio.run(main())
