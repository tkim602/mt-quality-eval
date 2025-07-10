#!/usr/bin/env python3
from pathlib import Path
import os, orjson
from tqdm import tqdm
from dotenv import load_dotenv
import anthropic                               

load_dotenv()                               
TAGGED_JSON = Path("out/tagged.json")
OUT_APE     = Path("out/ape_improved_prompt.json")

COMET_THRESH   = 0.80
CLAUDE_MODEL   = "claude-3-5-haiku-20241022"         
MAX_TOKENS_OUT = 256                              

client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

records = orjson.loads(TAGGED_JSON.read_bytes())

ape_targets = [
    r for r in records
    if r.get("labse_tag") == "fail" and r.get("comet_score", 1.0) < COMET_THRESH
]

system_prompt = (
    "You are a senior Korean <-> English *post editor*, not a translator.\n"
    "## Objective\n"
    "Start with the given machine translation (MT) as the **baseline**. "
    "Make only the edits needed to fix errors in meaning, terminology, grammar, punctuation, or style for better translation.\n"
    "## Hard Rules\n"
    "1. **Preserve placeholders and formatting** exactly\n"
    "2. **Keep every technical term that already appears in the MT** (e.g. 'crawling', 'firewall'). "
    "Replace a term only if it is demonstrably wrong *and* the Korean source proves it.\n"
    "3. Do **not** add, remove, or reorder CVE IDs, paths, numbers, or option names.\n"
    "4. Titles ≤ 20 words; descriptions ≤ 30 words when possible. If unsure which is title or description, refer to the given MT.\n"
    "5. Omit unnecessary articles (a/an/the) and pronouns—unless the MT already uses them.\n"
    "6. Maintain Korean UI words present in the MT (e.g. '탭', '모듈').\n"
    "## Output\n"
    "- Return **only** the post-edited English text.\n"
    "- Do not explain your change."
)

for r in tqdm(ape_targets, desc="progress"):
    user_prompt = (
        f"Post-edit:\n"
        f"Source (ko): {r['src']}\n"
        f"MT (en): {r['mt']}\n"
        "Corrected:"
    )

    response = client.messages.create(
        model=CLAUDE_MODEL,
        temperature=0,
        max_tokens=MAX_TOKENS_OUT,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    r["ape"] = response.content[0].text.strip()

OUT_APE.parent.mkdir(parents=True, exist_ok=True)
OUT_APE.write_bytes(orjson.dumps(ape_targets, option=orjson.OPT_INDENT_2))

print(f"{len(ape_targets)} items edited")
