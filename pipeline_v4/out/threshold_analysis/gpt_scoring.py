import json, os
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

INPUT_PATH  = Path(r"c:\Users\tkim602_global\Desktop\mt_eval\pipeline_v4\out\threshold_analysis\sample_for_manual_tagging3.json")
OUTPUT_PATH = Path("sample_tagged3.json")
MODEL       = "gpt-4o"         
TEMPERATURE = 0.6

load_dotenv()
client = OpenAI()   
system_prompt = {
    "role": "system",
    "content": (
        "You are a bilingual software localization QA expert performing MQM-style translation evaluation. "
        "You are fluent in Korean and English, and deeply familiar with software engineering, static analysis, and cybersecurity terminology.\n\n"
        "You are given a Korean source string (`src`) and its machine-translated English version (`mt`). For each pair, return strictly:\n\n"
        "- `Pass, reason` → if the translation is accurate, fluent, and appropriate for the context.\n"
        "- `Fail, reason` → if it contains any semantic, grammatical, or terminology issues.\n\n"
        "Output rules:\n"
        "- Be concise but human — write like a real translator reviewing work.\n"
        "- Do not rephrase or correct the sentence.\n"
        "- Keep the reason to a single sentence or short phrase.\n"
        "- No commentary, no markdown, no extra formatting.\n"
        "- Only return: `Pass, reason` or `Fail, reason`.\n"
    )
}

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def gpt_pass_fail(src: str, mt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[system_prompt, {"role": "user", "content": f"src: {src}\nmt: {mt}"}],
        temperature=TEMPERATURE
    )
    out = resp.choices[0].message.content.strip()
    return "Pass" if out.lower().startswith("pass") else "Fail"

data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

for rec in tqdm(data, desc="GPT evaluating"):
    if not rec.get("manual"):                
        try:
            rec["manual"] = gpt_pass_fail(rec["src"], rec["mt"])
        except Exception as e:
            rec["manual"] = "Fail"
            print("Error:", e)

OUTPUT_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\ndone")
