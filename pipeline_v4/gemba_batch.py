from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import List, Tuple

import orjson
from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError, APIError

import cfg

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _sys_prompt(n: int) -> str:
    return f"""You are a senior KO→EN rater for the quality of machine translation\n.
    Your task is to assess the quality of the translation.\n

### Score (0‑100, step 5)
• adequacy – fidelity & completeness to the source  
• fluency  – grammar, naturalness, English conventions  
• overall  = min(adequacy, fluency)

### Deductions (subtract from 100 / floor 0)
**ACCURACY**
-20 wrong meaning / full mistranslation  
-10 key term or placeholder omitted  
-5 nuance / secondary info missing  

**FLUENCY**
-15 major grammar / word‑order error  
-5 awkward phrasing, article misuse, passive voice abuse  
-5 casing / punctuation (typo always 0)  

### Evidence rule
overall ≥ 90 → evidence must return "perfect"  
else          → evidence = issue description for post editing
**Evidence** should identify the errors correctly for APE.

### Output
Return a JSON array of length {n}:  
[{{"idx":int, "overall":int, "adequacy":int, "fluency":int, "evidence":str}}, …]

If valid JSON cannot be produced, fall back to one plain‑text line per item:
idx,overall,adequacy,fluency,evidence
"""


def _messages(batch: List[dict]) -> List[dict]:
    payload = [
        {"idx": i + 1, "source": r["src"], "translation": r["mt"]}
        for i, r in enumerate(batch)
    ]
    return [
        {"role": "system", "content": _sys_prompt(len(batch))},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

_LINE_SPLIT = re.compile(r"[,:|\t\-]|")

def _parse_json(text: str, n: int):
    try:
        arr = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(arr, list):
        return None
    scores = [(0.0, 0.0, 0.0, "no_parse")] * n
    for item in arr:
        i = int(item.get("idx", 0)) - 1
        if 0 <= i < n:
            scores[i] = (
                float(item.get("overall", 0)),
                float(item.get("adequacy", 0)),
                float(item.get("fluency", 0)),
                str(item.get("evidence", "")).strip() or "ok",
            )
    return scores


def _parse_lines(text: str, n: int):
    scores = [(0.0, 0.0, 0.0, "no_parse")] * n
    for line in text.splitlines():
        if not line.strip():
            continue
        line = re.sub(r"^\s*(\d+)\s*[).]", r"\1", line)
        parts = re.split(r"[,:|\t\-]", line, maxsplit=4)
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0].strip()) - 1
            overall = float(parts[1])
            adequacy = float(parts[2])
            fluency = float(parts[3])
            evidence = parts[4].strip() if len(parts) > 4 else "ok"
        except ValueError:
            continue
        if 0 <= idx < n:
            scores[idx] = (overall, adequacy, fluency, evidence)
    return scores


def _parse(text: str, n: int):
    return _parse_json(text, n) or _parse_lines(text, n)

async def _ask(msgs: List[dict], retry: int = 5):
    for attempt in range(retry):
        try:
            resp = await client.chat.completions.create(
                model=cfg.GEMBA_MODEL,
                messages=msgs,
                temperature=0,
                max_tokens=1000,
            )
            return resp.choices[0].message.content
        except (RateLimitError, APIError) as e:
            wait = 2 ** attempt + random.random()
            logger.warning("%s — retrying in %.1f s", e.__class__.__name__, wait)
            await asyncio.sleep(wait)
    raise RuntimeError("OpenAI call failed after retries")


async def _score(batch: List[dict]):
    raw_text = await _ask(_messages(batch))
    return _parse(raw_text, len(batch))

def _decide(cos: float, comet: float, gemba: float, bucket: str):
    checks = [
        cos >= cfg.COS_THR[bucket],
        comet >= cfg.COMET_THR[bucket],
        gemba >= cfg.GEMBA_PASS,
    ]
    keys = ("cosine", "comet", "gemba")
    passed = [k for k, ok in zip(keys, checks) if ok]
    failed = [k for k in keys if k not in passed]
    # cos == 1은 {0}인 경우 edge case
    tag = "strict_pass" if (len(passed) == 3 or cos == 1.0) else "soft_pass" if len(passed) == 2 else "fail"
    return tag, passed, failed


def _ordered(rec: dict, tag: str, passed: List[str], failed: List[str]) -> dict:
    return {
        "key": rec["key"],
        "src": rec["src"],
        "mt": rec["mt"],
        "bucket": rec["bucket"],
        "cos": rec["cos"],
        "comet": rec["comet"],
        "gemba": rec["gemba"],
        "gemba_adequacy": rec["gemba_adequacy"],
        "gemba_fluency": rec["gemba_fluency"],
        "tag": tag,
        "flag": {"passed": passed, "failed": failed, "gemba_reason": rec.pop("_ev", "")},
    }

async def main():
    raw_path: Path = cfg.OUT_DIR / "filtered_v2.json"
    out_path: Path = cfg.OUT_DIR / "gemba_v3.json"

    raw: List[dict] = orjson.loads(raw_path.read_bytes())

    for i in range(0, len(raw), cfg.GEMBA_BATCH):
        batch = raw[i : i + cfg.GEMBA_BATCH]
        logger.info("Scoring items %d–%d …", i + 1, i + len(batch))
        for rec, (ov, adq, flu, ev) in zip(batch, await _score(batch)):
            rec["gemba"] = ov
            rec["gemba_adequacy"] = adq
            rec["gemba_fluency"] = flu
            rec["_ev"] = ev

    final = []
    for rec in raw:
        tag, p, f = _decide(rec["cos"], rec["comet"], rec["gemba"], rec["bucket"])
        final.append(_ordered(rec, tag, p, f))

    out_path.write_bytes(orjson.dumps(final, option=orjson.OPT_INDENT_2))
    logger.info("Done → %s", out_path)


if __name__ == "__main__":
    asyncio.run(main())
