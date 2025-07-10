from pathlib import Path
import os, orjson, openai
from tqdm import tqdm

TAGGED_JSON = Path("out/tagged.json")
# TERM_BASE_JSON = Path("../samples/term_base_map.json")
# OUT_TERM = Path("out/term_checked.json")
OUT_APE = Path("out/ape_improved_prompt.json") 

openai.api_key = os.getenv("OPENAI_API_KEY")

# 괜찮은 걸 찾아야할 것 같음 -> cosine similarity를 strict하게하면 comet은 적당해도 괜찮을 것 같기도함
COMET_THRESH = 0.80

# raw = orjson.loads(TERM_BASE_JSON.read_bytes())
# term_base = {}
# if isinstance(raw, dict):
#     tb_map = raw
# else:
#     for item in raw:
#         if isinstance(item, dict):
#             en = (
#                 item.get("en")
#                 or item.get("en-US")
#                 or item.get("en_US")
#                 or item.get("en_term")
#             )
#             ko = (
#                 item.get("ko")
#                 or item.get("ko-KR")
#                 or item.get("ko_term")
#             )
#             if en and ko:
#                 term_base[en] = ko
#         elif isinstance(item, (list, tuple)) and len(item) >= 2:
#             term_base[item[0]] = item[1]
#         elif isinstance(item, str) and ":" in item:
#             en, ko = map(str.strip, item.split(":", 1))
#             term_base[en] = ko

records = orjson.loads(TAGGED_JSON.read_bytes())

for r in records:
    src, mt = r["src"], r["mt"]
    # missing = [
    #     en for en, ko in tb_map.items()
    #     if (ko in src and en not in mt) or (en in mt and ko not in src)
    # ]
    # r["term_check"] = "pass" if not missing else "fail"
    # if missing:
    #     r["missing_terms"] = missing

# OUT_TERM.parent.mkdir(parents=True, exist_ok=True)
# OUT_TERM.write_bytes(orjson.dumps(records, option=orjson.OPT_INDENT_2))

# cosine similarity는 낮은데 comet 점수가 높으면 혹시 UI적인 요소나 다른 요인이 있을 수
# 있을 것 같아서, 우선 제외하도록했습니다. 
ape_targets = [
    r for r in records if r.get("labse_tag") == "fail" 
    and r.get("comet_score", 1.0) < COMET_THRESH
]

system_prompt = (
    "You are a senior Korean <-> English *post editor*, not a translator."
    "## Objective"
    "Start with the given machine translation (MT) as the **baseline**."
    "Make only the edits needed to fix errors in meaning, terminology, grammar, punctuation, or style for better translation."
    
    "## Hard Rules"
    "1. **Preserve placeholders and formatting** exactly"
    "2. **Keep every technical term that already appears in the MT** (e.g. 'crawling', 'firewall')."
    "   - Replace a term only if it is demonstrably wrong *and* the Korean source proves it."
    "3. Do **not** add, remove, or reorder CVE IDs, paths, numbers, or option names."
    "4. Titles <= 20 words; descriptions <= 30 words when possible. If not sure which is title or description, refer to the given MT."
    "5. Omit unecessary articles (a/an/the) and pronouns."
    "   - But if the given machine translation is using pronouns, you do not have to forcbly remove the pronouns."
    "   - Once again, it is important to *post-edit*, not generatig a creative translation every time."
    "6. Maintain Korean UI words that appear in the given MT (e.g. '탭', '모듈')"

    "## Output"
    "- Return **only** the post-edited English text."
    "- Do not explain your change."
)

for r in tqdm(ape_targets, desc="progress"):                    
    user_prompt = (
        f"Post-edit:\n"
        f"Source (ko): {r['src']}\n"
        f"MT (en): {r['mt']}\n"
        "Corrected:"
    )
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    r["ape"] = response.choices[0].message.content.strip()

OUT_APE.write_bytes(orjson.dumps(ape_targets, option=orjson.OPT_INDENT_2))

print(f"{len(ape_targets)} items edited")
