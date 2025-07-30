# gemba_batch_multistyle.py
# Multi‑Style GEMBA Evaluation Module (formal, news, casual, literature, poetry)

from __future__ import annotations

import asyncio, json, logging, os, random, re
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import orjson
from openai import AsyncOpenAI, RateLimitError, APIError
from dotenv import load_dotenv

import cfg

# ───────────────────────────[ 환경 & 로깅 ]───────────────────────────────────
load_dotenv()
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ───────────────────────────[ 1. STYLE_PATTERNS ]────────────────────────────
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

# ───────────────────────────[ 2. Style Detector ]────────────────────────────
def detect_text_style(text: str) -> str:
    scores = {s: 0 for s in STYLE_PATTERNS}
    for style, pats in STYLE_PATTERNS.items():
        for pat in pats:
            hits = len(re.findall(pat, text, flags=re.IGNORECASE))
            weight = 3 if style == "casual" else 2.5 if style == "news" else 2 if style == "formal" else 1
            scores[style] += hits * weight
    if len(text) < 25:
        scores["casual"] += 4
    if len(text) > 120 and re.search(r"(습니다|됩니다|입니다)", text):
        scores["formal"] += 4
    if re.search(r"(발표|보도|밝혔|전했|조사|정부)", text):
        scores["news"] += 5
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "formal"

# ───────────────────────────[ 3. GEMBA System Prompts ]──────────────────────
def _sys_prompt_formal(n: int, vs: str) -> str:  # 이하 5개 프롬프트 동일 형식
    return f"""You are an expert Korean→English translation assessor … (생략) …

### Validation Context
{vs}
### Output Format
JSON array of exactly {n} items …"""

def _sys_prompt_news(n: int, vs: str) -> str:    # (본문 생략)
    return f"...{vs}..."

def _sys_prompt_casual(n: int, vs: str) -> str:  # (본문 생략)
    return f"...{vs}..."

def _sys_prompt_literature(n: int, vs: str) -> str:
    return f"...{vs}..."

def _sys_prompt_poetry(n: int, vs: str) -> str:
    return f"...{vs}..."

PROMPT_MAP = {
    "formal": _sys_prompt_formal,
    "news": _sys_prompt_news,
    "casual": _sys_prompt_casual,
    "literature": _sys_prompt_literature,
    "poetry": _sys_prompt_poetry,
}

def get_style_prompt(style: str, n: int, vs: str) -> str:
    return PROMPT_MAP.get(style, _sys_prompt_formal)(n, vs)

# ───────────────────────────[ 4. _messages Builder ]─────────────────────────
def _messages(batch: List[dict], style: str) -> List[dict]:
    # validation 요약
    vrep = []
    for i, r in enumerate(batch):
        val = r.get("validation", {})
        s = f"  Item {i+1}:\n"
        if val.get("term_consistency", {}).get("score", 1.0) < 1.0:
            s += "    - Terminology mismatch\n"
        if not val.get("placeholder_check", {}).get("passed", True):
            s += "    - Placeholder integrity fail\n"
        if val.get("readability_score", 100) < 50:
            s += f"    - Low readability {val['readability_score']:.0f}\n"
        if s == f"  Item {i+1}:\n":
            s += "    - All automated checks passed\n"
        vrep.append(s)
    vsum = "### Validation Summary\n" + "".join(vrep)

    # 아주 짧은 스타일별 예시 1개만 삽입 (생략 가능)
    ex = {"formal": "### Examples\nKorean: A\nEnglish: A",
          "news": "### Examples\nKorean: B\nEnglish: B",
          "casual": "### Examples\nKorean: C\nEnglish: C",
          "literature": "### Examples\nKorean: D\nEnglish: D",
          "poetry": "### Examples\nKorean: E\nEnglish: E"}[style]

    items = [{"idx": i + 1, "source": r["src"], "translation": r["mt"]} for i, r in enumerate(batch)]
    return [
        {"role": "system", "content": get_style_prompt(style, len(batch), vsum)},
        {"role": "user", "content": ex + "\n\n### Items to Evaluate\n" + json.dumps(items, ensure_ascii=False)},
    ]

# ───────────────────────────[ 5. 파서 · _ask · _score ]──────────────────────
_LINE_SPLIT = re.compile(r"[,:|\t\-]|")

def _parse_json(text: str, n: int):
    try:
        arr = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(arr, list):
        return None
    res = [(0.0, 0.0, 0.0, "no_parse")] * n
    for it in arr:
        i = int(it.get("idx", 0)) - 1
        if 0 <= i < n:
            res[i] = (float(it.get("overall", 0)), float(it.get("adequacy", 0)),
                      float(it.get("fluency", 0)), str(it.get("evidence", "")).strip() or "ok")
    return res

def _parse_lines(text: str, n: int):
    res = [(0.0, 0.0, 0.0, "no_parse")] * n
    for ln in text.splitlines():
        if not ln.strip():
            continue
        ln = re.sub(r"^\s*(\d+)\s*[).]", r"\1", ln)
        parts = re.split(_LINE_SPLIT, ln, maxsplit=4)
        if len(parts) < 4:
            continue
        try:
            i = int(parts[0]) - 1
            res[i] = (float(parts[1]), float(parts[2]), float(parts[3]), parts[4].strip() if len(parts) > 4 else "ok")
        except ValueError:
            pass
    return res

def _parse(text: str, n: int):
    return _parse_json(text, n) or _parse_lines(text, n) or [(0.0, 0.0, 0.0, "no_parse")] * n

async def _ask(msgs: List[dict], retry: int = 8):
    for a in range(retry):
        try:
            r = await client.chat.completions.create(
                model=cfg.GEMBA_MODEL, messages=msgs,
                temperature=0.0, max_tokens=2000, top_p=0.9)
            return r.choices[0].message.content
        except (RateLimitError, APIError):
            await asyncio.sleep(min(60, 2 ** a + random.random()))
    raise RuntimeError("OpenAI call failed")

async def _score(batch: List[dict], style: str):
    raw = await _ask(_messages(batch, style))
    return _parse(raw, len(batch))

# ───────────────────────────[ 6. 외부 호출 함수 ]────────────────────────────
async def gemba_batch_multistyle(src: List[str], tgt: List[str], style: str | None = None):
    if len(src) != len(tgt):
        raise ValueError("src & tgt length mismatch")
    style = style or detect_text_style(src[0])
    bs, res, batch = cfg.GEMBA_BATCH, [], [{"src": s, "mt": t} for s, t in zip(src, tgt)]
    for i in range(0, len(batch), bs):
        res.extend(await _score(batch[i:i + bs], style))
    return res

# ───────────────────────────[ 7. 메인 파이프라인 ]───────────────────────────
def _decide(cos: float, comet: float, g: float, buck: str):
    ok = [cos >= cfg.COS_THR[buck], comet >= cfg.COMET_THR[buck], g >= cfg.GEMBA_PASS]
    k = ("cosine", "comet", "gemba")
    tag = "strict_pass" if all(ok) or cos == 1.0 else "soft_pass" if sum(ok) == 2 else "fail"
    return tag, [x for x, v in zip(k, ok) if v], [x for x, v in zip(k, ok) if not v]

async def main():
    run = Path(os.getenv("RUN_DIR", cfg.OUT_DIR))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    in_f = (run / cfg.FILTER_OUTPUT_FILENAME) if os.getenv("RUN_DIR") else run / f"filtered_{ts}.json"
    out_f = run / (cfg.GEMBA_OUTPUT_FILENAME.replace(".json", "_multistyle.json") if os.getenv("RUN_DIR")
                   else f"gemba_multistyle_{ts}.json")
    if not in_f.exists():
        logger.error("Input not found: %s", in_f); return
    raw: List[dict] = orjson.loads(in_f.read_bytes())
    sty_grp: Dict[str, List[int]] = {s: [] for s in STYLE_PATTERNS}
    for i, r in enumerate(raw):
        st = detect_text_style(r["src"]); r["detected_style"] = st; sty_grp[st].append(i)
    sem = asyncio.Semaphore(4)

    async def worker(sty: str, idxs: List[int]):
        async with sem:
            bs = cfg.GEMBA_BATCH
            for s in range(0, len(idxs), bs):
                b = [raw[i] for i in idxs[s:s + bs]]
                sc = await _score(b, sty)
                for i, (ov, ad, fl, ev) in zip(idxs[s:s + bs], sc):
                    raw[i].update(gemba=ov, gemba_adequacy=ad, gemba_fluency=fl, _ev=ev)

    await asyncio.gather(*(worker(st, idxs) for st, idxs in sty_grp.items() if idxs))

    final = []
    for r in raw:
        tag, p, f = _decide(r["cos"], r["comet"], r["gemba"], r["bucket"])
        final.append({
            "key": r["key"], "src": r["src"], "mt": r["mt"], "bucket": r["bucket"],
            "cos": r["cos"], "comet": r["comet"], "gemba": r["gemba"],
            "gemba_adequacy": r["gemba_adequacy"], "gemba_fluency": r["gemba_fluency"],
            "tag": tag, "flag": {"passed": p, "failed": f, "gemba_reason": r.pop("_ev", "")},
            "validation": r.get("validation", {}), "detected_style": r["detected_style"],
        })

    out_f.write_bytes(orjson.dumps(final, option=orjson.OPT_INDENT_2))
    logger.info("Done → %s", out_f)

if __name__ == "__main__":
    asyncio.run(main())
