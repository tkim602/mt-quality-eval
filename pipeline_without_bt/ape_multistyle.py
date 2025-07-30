# ape_multistyle.py
# Multi‑Style Automatic Post‑Editing (formal, news, casual, literature, poetry)

from __future__ import annotations

import asyncio, json, logging, os, random, re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np, orjson, torch
from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError, APIError
from sentence_transformers import SentenceTransformer
from tqdm.asyncio import tqdm_asyncio

import cfg

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------------------------------------------------------------------------- #
# 1) STYLE_PATTERNS (GEMBA와 동일)                                            #
# --------------------------------------------------------------------------- #
STYLE_PATTERNS: Dict[str, List[str]] = {
    "formal": [
        r"에 (대한|관한)|관련(?:하여|된)|비하여|따라서|그러므로|이에 따라|및|등(?:의)?|로써",
        r"사용자|시스템|데이터(?:베이스)?|오류|서버|클라이언트|프로토콜|알고리즘|모듈|인증|권한|정책|설정|프로세스|요청|응답|스펙|성능",
        r"합니다|합니다만|습니다|됩니다|입니다|되어 있습니다|고 있습니다|것입니다|될 수 있습니다",
    ],
    "news": [
        r"기자|뉴스|보도|속보|발(?:표|사)|전했다|밝혔다|언론|취재|인터뷰|보도자료|논평|성명",
        r"정부|대통령|총리|장관|국회|의회|시장|지사|경찰|검찰|법원|국방부|통계청|기상청|질병관리청",
        r"오늘|어제|내일|금일|지난|다음|이번|올해|작년|내년|최근|방금",
        r"(%|억|만|천)원|달러|유로|위안|도|℃|㎜|km|㎞|명",
        r"사망|부상|확진|감염|집계|총선|대선|투표|여론조사|승인|제정|통과",
    ],
    "casual": [
        r"ㅋㅋ+|ㅎㅎ+|ㅠ+|ㅜ+|ㅎㄷㄷ|헐|헉|OMG|lol|LOL|rofl|lmao|ㄱㄱ|ㄴㄴ|ㄷㄷ|ㄹㅇ|ㅇㅇ|ㅇㅋ|ㅇㅈ",
        r"야|너|나|우리|걔|쟤|얘|니가|내가|진짜|완전|대박|레알|졸라|존맛|쩐다|핵꿀|꿀잼|노잼|뭐야|뭔데",
        r"했어|했냐|했네|할래|할게|할껀데|될까|해야지|싶다|같아|인가|거야|거지|ㄹ까",
    ],
    "literature": [
        r"그의|그녀(?:의)?|소년|소녀|노인|작은|커다란|깊은|푸른|붉은|검은|창백한|희미한|은은한",
        r"마음|영혼|운명|사랑|고독|슬픔|기쁨|한숨|그늘|빛|어둠|환희|절망|추억|회상|울림",
        r"하늘|바다|강|호수|숲|계곡|꽃|나무|별|달|구름|바람|비|눈|햇살|노을|파도",
        r"였(?:다|네|구나)|이었(?:다|네)|하였다|하였으나|하노라|하리라|하더라|물었다|속삭였다|외쳤다",
    ],
    "poetry": [
        r"님|그대|그리움|마음|한숨|눈물|별빛|달빛|바람결|새벽|노래|향기|고요|적막|설렘|울림",
        r"바람|꽃|달|별|하늘|구름|물결|산들|강|숲|비|눈|파도|새|물|시냇물",
        r"하네|하노|하구나|로다|이로다|이여|이니라|도다|누나|노라|이련가|어라",
        r"\/|\n",
    ],
}

# --------------------------------------------------------------------------- #
# 2) 가중치 style detector                                                    #
# --------------------------------------------------------------------------- #
def detect_text_style(text: str) -> str:
    scores = {s: 0 for s in STYLE_PATTERNS}
    for style, pats in STYLE_PATTERNS.items():
        for pat in pats:
            hits = len(re.findall(pat, text, flags=re.IGNORECASE))
            weight = 3 if style == "casual" else 2.5 if style == "news" else 2 if style == "formal" else 1
            scores[style] += hits * weight

    length = len(text)
    if length < 25:
        scores["casual"] += 4
    if length > 120 and re.search(r"(습니다|됩니다|입니다)", text):
        scores["formal"] += 4
    if re.search(r"(발표|보도|밝혔|전했|조사|정부)", text):
        scores["news"] += 5

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "formal"

# --------------------------------------------------------------------------- #
# 3) 나머지 원본 APE 로직 (프롬프트 함수, _call_gpt, cosine/comet, main 등)     #
# --------------------------------------------------------------------------- #

def load_termbase():
    if hasattr(cfg, "TERMBASE_PATH") and Path(cfg.TERMBASE_PATH).exists():
        with open(cfg.TERMBASE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return {it["ko"]: it["en-US"] for it in data if "ko" in it and "en-US" in it}
            return data
    return {}

TERM = ", ".join(f"{k}→{v}" for k, v in load_termbase().items())

# ---------- get_style_prompt (원본 그대로) -----------------------------------
# ... (formal_prompts / news_prompts / casual_prompts / literature_prompts / poetry_prompts)
# --------------------------------------------------------------------------- #

SEM = asyncio.Semaphore(getattr(cfg, "APE_CONCURRENCY", 8))

async def _call_gpt(prompt: str, mode: str, style: str, retry: int = 8):
    async with SEM:
        temp_map = {"formal": (0.1, 0.3), "news": (0.15, 0.35),
                    "casual": (0.2, 0.4), "literature": (0.25, 0.45),
                    "poetry": (0.3, 0.5)}
        temperature = temp_map.get(style, (0.1, 0.3))[0 if mode == "soft_pass" else 1]
        for a in range(retry):
            try:
                resp = await client.chat.completions.create(
                    model=cfg.APE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature)
                return resp.choices[0].message.content.strip()
            except (RateLimitError, APIError):
                await asyncio.sleep(min(60, 2 ** a + random.uniform(1, 3)))
        return "[APE_FAILED_RATE_LIMIT]"

async def edit_sentence_multistyle(src: str, mt: str, mode: str, ev: str, style: str | None):
    style = style or detect_text_style(src)
    prompt = get_style_prompt(style, mode, ev) + f"\n\nSRC (ko): {src}\nMT  (en): {mt}"
    return await _call_gpt(prompt, mode, style)

# ---------- cosine / comet helpers ------------------------------------------
_labse = SentenceTransformer(getattr(cfg, "COS_MODEL", "sentence-transformers/LaBSE")).to(getattr(cfg, "DEVICE", "cpu"))
try:
    from comet import load_from_checkpoint, download_model
    _comet = load_from_checkpoint(download_model(getattr(cfg, "COMET_CKPT", "Unbabel/wmt22-cometkiwi-da"))).to(getattr(cfg, "DEVICE", "cpu"))
    _comet.eval()
    _has_comet = True
except Exception:
    _has_comet = False

@torch.no_grad()
def cosine_batch(src: List[str], tgt: List[str]):
    if not src:
        return np.empty(0, np.float32)
    a = _labse.encode(src, convert_to_tensor=True, normalize_embeddings=True, batch_size=32)
    b = _labse.encode(tgt, convert_to_tensor=True, normalize_embeddings=True, batch_size=32)
    return (a * b).sum(-1).cpu().numpy()

def comet_batch(src: List[str], tgt: List[str]):
    if not _has_comet:
        return [float("nan")] * len(src)
    data = [{"src": s, "mt": t} for s, t in zip(src, tgt)]
    preds = _comet.predict(data, batch_size=16, gpus=0, progress_bar=True)
    return [float(x) for x in preds["scores"]]

# ----------------------------- main -----------------------------------------
async def main():
    run_dir = os.getenv("RUN_DIR")
    if run_dir:
        run = Path(run_dir)
        in_file = cfg.GEMBA_OUTPUT_FILENAME.replace(".json", "_multistyle.json") if "multistyle" in str(run) else cfg.GEMBA_OUTPUT_FILENAME
        out_file = cfg.APE_OUTPUT_FILENAME.replace(".json", "_multistyle.json")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run = cfg.OUT_DIR
        in_file = f"gemba_multistyle_{ts}.json"
        out_file = f"ape_multistyle_{ts}.json"

    in_path, out_path = run / in_file, run / out_file
    if not in_path.exists():
        logger.error("Input not found: %s", in_path)
        return

    items: List[dict] = orjson.loads(in_path.read_bytes())
    targets = [i for i, r in enumerate(items) if r.get("tag") in ("soft_pass", "fail")]

    async def process(idx: int):
        r = items[idx]
        ev = r.get("flag", {}).get("gemba_reason", "n/a")
        style = r.get("detected_style", "formal")
        r["ape"] = await edit_sentence_multistyle(r["src"], r["mt"], r["tag"], ev, style)
        r["ape_style"] = style

    await tqdm_asyncio.gather(*(process(i) for i in targets), total=len(targets))
    ape_idx = [i for i in targets if "ape" in items[i]]

    src = [items[i]["src"] for i in ape_idx]
    ape = [items[i]["ape"] for i in ape_idx]
    ape_cos, ape_com = cosine_batch(src, ape), comet_batch(src, ape)

    for i, c, m in zip(ape_idx, ape_cos, ape_com):
        it = items[i]
        it["ape_cos"], it["ape_comet"] = float(c), float(m)
        it["delta_cos"], it["delta_comet"] = float(c - it.get("cos", 0.0)), float(m - it.get("comet", 0.0))

    from gemba_batch_multistyle import gemba_batch_multistyle
    style_groups: Dict[str, List[int]] = {}
    for i in ape_idx:
        style_groups.setdefault(items[i]["ape_style"], []).append(i)

    for st, idxs in style_groups.items():
        s_src = [items[i]["src"] for i in idxs]
        s_ape = [items[i]["ape"] for i in idxs]
        g_scores = await gemba_batch_multistyle(s_src, s_ape, st)
        for i, g in zip(idxs, g_scores):
            g_val = float(g[0]) if isinstance(g, tuple) else float(g.get("overall", 0.0))
            it = items[i]
            it["ape_gemba"] = g_val
            it["delta_gemba"] = g_val - it.get("gemba", 0.0)

    final: List[dict] = []
    for r in items:
        od = OrderedDict()
        for key in ("key", "src", "mt", "ape", "ape_style", "detected_style",
                    "validation", "ape_cos", "delta_cos", "ape_comet", "delta_comet",
                    "ape_gemba", "delta_gemba"):
            if key in r:
                od[key] = r[key]
        for k, v in r.items():
            if k not in od:
                od[k] = v
        final.append(od)

    out_path.write_bytes(orjson.dumps(final, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    logger.info("Done → %s", out_path)

if __name__ == "__main__":
    asyncio.run(main())
