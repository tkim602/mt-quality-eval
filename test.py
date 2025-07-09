from dotenv import load_dotenv
import openai
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

source = "업데이트 후 시스템이 자동으로 재시작됩니다."
reference = "The system will automatically restart after the update."
translation = "The system will reboot on its own after updating."

def build_prompt(source=None, reference=None, translation=None):
    parts = [
        "Score the following translation from Korean to English",
        "on a continuous scale from 0 to 100 that starts on \"No meaning preserved\",",
        "goes through \"Some meaning preserved\", then \"Most meaning preserved and few grammar mistakes\",",
        "up to \"Perfect meaning and grammar\".",
        "",
        "Respond with a score followed by a brief reason.",
    ]
    if source:
        parts.append(f'Korean source: "{source}"')
    if reference:
        parts.append(f'English human reference: "{reference}"')
    if translation:
        parts.append(f'English translation: "{translation}"')
    parts.append("\nScore (0–100) and reason:")
    return "\n".join(parts)

def run_all(source, reference, translation, model="gpt-4"):
    modes = {
        "T": build_prompt(source=None, reference=None, translation=translation),
        "S-T": build_prompt(source=source, reference=None, translation=translation),
        "R-T": build_prompt(source=None, reference=reference, translation=translation),
        "S-R-T": build_prompt(source=source, reference=reference, translation=translation),
    }

    results = {}
    for mode, prompt in modes.items():
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100,
        )
        text = response.choices[0].message["content"].strip()
        try:
            score = int(text.split()[0])
        except:
            score = None
        results[mode] = {"score": score, "raw": text}
    return results

# Run and print results
results = run_all(source, reference, translation)
for mode, r in results.items():
    print(f"{mode}: {r['score']} ({r['raw']})")
