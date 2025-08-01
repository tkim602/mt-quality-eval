from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
from datetime import datetime
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

def _sys_prompt(n: int, validation_summary: str) -> str:
    return f"""You are an expert Korean→English translation quality assessor with deep understanding of both languages, technical terminology, and slangs, etc.

### Assessment Criteria (0–100, step 5)
**ADEQUACY** – Source fidelity & completeness
- Does the translation convey the exact meaning?
- Are all key concepts preserved?
- Is technical terminology correctly translated?

**FLUENCY** – Target language quality
- Is the English natural and grammatically correct?
- Does it follow English conventions?
- Is it readable for native speakers?

**OVERALL** = min(adequacy, fluency)

### Scoring Penalties (subtract from 100, floor at 0)
**CRITICAL ERRORS (-15 to -25)**
- Complete mistranslation or opposite meaning
- Missing critical information or key terms
- Severe grammar errors that impede understanding

**MAJOR ERRORS (-10 to -15)**
- Significant meaning shifts or omissions
- Important terminology inconsistencies
- Major grammatical issues (word order, tense, etc.)

**MINOR ERRORS (-3 to -8)**
- Nuance loss or awkward phrasing
- Minor terminology issues
- Article errors, preposition mistakes
- Punctuation or formatting problems

### Korean→English Specific Considerations
- **Honorifics**: Appropriate handling of Korean politeness levels
- **Technical Terms**: Consistent use of established terminology
- **Cultural Context**: Proper localization for English speakers
- **Sentence Structure**: Natural English flow vs. Korean structure

### Validation Context
{validation_summary}

### Evidence Guidelines (IMPORTANT: Provide evidence in Korean)
- Score ≥90: "완벽함" 또는 "우수함"과 같은 간단한 한국어 피드백
- Score 70-89: 주요 개선점에 대한 간단한 한국어 설명
- Score <70: 상세한 오류 분석을 한국어로 제공 (포스트 에디팅용)

### Evidence Format Requirements
- Write ALL evidence in Korean (한국어로 증거 작성)
- When mentioning specific problematic English terms, put them in quotes
- Example: "Critical"이라는 단어가 불분명합니다
- Example: 문법 오류가 있습니다: "was occurred" 대신 "occurred"를 사용해야 합니다
- Example: 용어 일관성 문제: "vulnerability" 번역이 일관되지 않습니다

### Output Format
JSON array of exactly {n} items:
[{{"idx":int, "overall":int, "adequacy":int, "fluency":int, "evidence":str}}, ...]

Fallback format if JSON fails:
idx,overall,adequacy,fluency,evidence
"""


def _messages(batch: List[dict]) -> List[dict]:
    validation_reports = []
    for i, r in enumerate(batch):
        val = r.get("validation", {})
        report = f"  Item {i+1}:\n"
        if val.get("term_consistency", {}).get("score", 1.0) < 1.0:
            mismatches = val["term_consistency"]["mismatches"]
            report += f"    - Terminology: {len(mismatches)} mismatch(es). e.g., '{mismatches[0]['src_term']}' -> expected '{mismatches[0]['expected_mt']}'.\n"
        if not val.get("placeholder_check", {}).get("passed", True):
            report += f"    - Placeholders: Integrity check failed.\n"
        if val.get("readability_score", 100) < 50:
            report += f"    - Readability: Score is {val['readability_score']:.0f}, may be hard to read.\n"
        if report == f"  Item {i+1}:\n":
            report += "    - All automated checks passed.\n"
        validation_reports.append(report)
    
    validation_summary = "### Validation Summary\n" + "".join(validation_reports)

    examples = """
### Examples for Reference (한국어 증거 제공 예시)

Example 1:
Korean: "사용자 인증이 실패했습니다."
English: "User authentication failed."
Assessment: overall=90, adequacy=95, fluency=85, evidence="완벽한 의미 전달과 자연스러운 영어 표현"

Example 2:
Korean: "데이터베이스 연결 오류가 발생했습니다."
English: "Database connection error was occurred."
Assessment: overall=70, adequacy=85, fluency=70, evidence="의미는 명확하지만 문법 오류: "was occurred" 대신 "occurred" 또는 "has occurred"를 사용해야 합니다"

Example 3:
Korean: "시스템이 정상적으로 종료되었습니다."
English: "The system was shutdown normally."
Assessment: overall=75, adequacy=80, fluency=75, evidence="소소한 문제들: "shutdown"은 동사형인 "shut down"이어야 하고, "normally" 대신 "successfully"가 더 자연스럽습니다"

Example 4:
Korean: "중요 변수를 public에 선언했습니다."
English: "critical variable in public field."
Assessment: overall=60, adequacy=65, fluency=55, evidence="심각한 문제들: "critical variable in public field"는 불완전한 문장이고 맥락이 부족합니다. "Declared important variables as public"과 같이 완전한 문장으로 표현해야 합니다"
"""

    payload = [
        {"idx": i + 1, "source": r["src"], "translation": r["mt"]}
        for i, r in enumerate(batch)
    ]
    
    return [
        {"role": "system", "content": _sys_prompt(len(batch), validation_summary)},
        {"role": "user", "content": examples + "\n\n### Items to Evaluate\n" + json.dumps(payload, ensure_ascii=False)},
    ]

_LINE_SPLIT = re.compile(r"[,:|\t\-]|")

def _parse_json(text: str, n: int):
    try:
        arr = json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"JSON parsing failed for text: {text[:200]}...")
        return None
    if not isinstance(arr, list):
        logger.warning(f"Parsed JSON is not a list: {type(arr)}")
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
    result = _parse_json(text, n) or _parse_lines(text, n)
    if result is None:
        logger.warning(f"Both JSON and line parsing failed for text: {text[:200]}...")
    else:
        zero_count = sum(1 for score in result if score[0] == 0.0)
        if zero_count > 0:
            logger.warning(f"Found {zero_count}/{n} zero scores in parsing result")
            logger.debug(f"Raw text: {text}")
    return result

async def _ask(msgs: List[dict], retry: int = 10):
    for attempt in range(retry):
        try:
            resp = await client.chat.completions.create(
                model=cfg.GEMBA_MODEL,
                messages=msgs,
                temperature=0.0, 
                max_tokens=2000, 
                top_p=0.9,      
            )
            return resp.choices[0].message.content
        except (RateLimitError, APIError) as e:
            wait = min(2 ** attempt + random.random(), 60)  
            logger.warning("%s — retrying in %.1f s (attempt %d/%d)", e.__class__.__name__, wait, attempt + 1, retry)
            await asyncio.sleep(wait)
        except Exception as e:
            logger.error(f"Unexpected error in GEMBA API call: {e}")
            wait = min(2 ** attempt + random.random(), 30)
            logger.warning(f"Retrying in %.1f s due to unexpected error", wait)
            await asyncio.sleep(wait)
    raise RuntimeError("OpenAI call failed after retries")


async def _score(batch: List[dict]):
    raw_text = await _ask(_messages(batch))
    logger.debug(f"GEMBA raw response for batch of {len(batch)}: {raw_text[:500]}...")
    result = _parse(raw_text, len(batch))
    if result is None:
        logger.error(f"Failed to parse GEMBA response for batch of {len(batch)} items")
        logger.debug(f"Full raw text: {raw_text}")
        return [(0.0, 0.0, 0.0, "no_parse")] * len(batch)
    return result

async def gemba_batch(src_texts: List[str], target_texts: List[str]) -> List[dict]:

    if len(src_texts) != len(target_texts):
        raise ValueError("Source and target text lists must have the same length")
    
    batch_data = []
    for i, (src, tgt) in enumerate(zip(src_texts, target_texts)):
        batch_data.append({
            "key": f"batch_item_{i}",
            "src": src,
            "mt": tgt 
        })
    results = []
    batch_size = cfg.GEMBA_BATCH
    
    for i in range(0, len(batch_data), batch_size):
        batch = batch_data[i:i + batch_size]
        scores = await _score(batch)
        results.extend(scores)
    
    return results

def _decide(cos: float, comet: float, gemba: float, bucket: str):
    checks = [
        cos >= cfg.COS_THR[bucket],
        comet >= cfg.COMET_THR[bucket],
        gemba >= cfg.GEMBA_PASS,
    ]
    keys = ("cosine", "comet", "gemba")
    passed = [k for k, ok in zip(keys, checks) if ok]
    failed = [k for k in keys if k not in passed]
    tag = "strict_pass" if (len(passed) == 3 or cos == 1.0) else "soft_pass" if len(passed) == 2 else "fail"
    return tag, passed, failed


def _ordered(rec: dict, tag: str, passed: List[str], failed: List[str]) -> dict:
    validation_data = rec.get("validation", {})
    ordered_rec = {
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
        "validation": validation_data,
        "domain": rec.get("domain", "sparrow"),  # Preserve domain info
    }
    return ordered_rec

async def main():
    run_dir = os.getenv('RUN_DIR')
    if run_dir:
        run_path = Path(run_dir)
        input_filename = cfg.FILTER_OUTPUT_FILENAME
        output_filename = cfg.GEMBA_OUTPUT_FILENAME
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_path = cfg.OUT_DIR
        input_filename = f"filtered_{timestamp}.json"
        output_filename = f"gemba_{timestamp}.json"
    
    raw_path: Path = run_path / input_filename
    out_path: Path = run_path / output_filename

    if not raw_path.exists():
        logger.error(f"Input file not found: {raw_path}")
        return

    raw: List[dict] = orjson.loads(raw_path.read_bytes())

    semaphore = asyncio.Semaphore(4)  
    
    async def process_batch(batch_start: int):
        async with semaphore:
            batch = raw[batch_start : batch_start + cfg.GEMBA_BATCH]
            logger.info("Scoring items %d–%d …", batch_start + 1, batch_start + len(batch))
            scores = await _score(batch)
            for rec, (ov, adq, flu, ev) in zip(batch, scores):
                rec["gemba"] = ov
                rec["gemba_adequacy"] = adq
                rec["gemba_fluency"] = flu
                rec["_ev"] = ev

    batch_starts = list(range(0, len(raw), cfg.GEMBA_BATCH))
    
    from tqdm.asyncio import tqdm as tqdm_asyncio
    await tqdm_asyncio.gather(
        *(process_batch(i) for i in batch_starts),
        desc="GEMBA batches"
    )

    final = []
    for rec in raw:
        tag, p, f = _decide(rec["cos"], rec["comet"], rec["gemba"], rec["bucket"])
        final.append(_ordered(rec, tag, p, f))

    out_path.write_bytes(orjson.dumps(final, option=orjson.OPT_INDENT_2))
    logger.info("Done → %s", out_path)


if __name__ == "__main__":
    asyncio.run(main())
