from pathlib import Path
import os, orjson, numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from comet import download_model, load_from_checkpoint
import anthropic
from dotenv import load_dotenv

load_dotenv()
APE_JSON = Path("out/ape_improved_prompt.json")
OUT_JSON = Path("out/ape_eval.json")

client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
embed_model = SentenceTransformer("sentence-transformers/LaBSE")
comet = load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da"))

summary = orjson.loads(APE_JSON.read_bytes())

def back_translate(en_text: str) -> str:
    resp = client.messages.create(
        model="claude-3-5-haiku-20241022",
        system=(
            "Translate the following text from English to Korean. "
            "This is to verify the accuracy of the initial Korean→English translation."
        ),
        messages=[{"role": "user", "content": en_text}],
        max_tokens=256,
        temperature=0,
    )
    return resp.content[0].text.strip()

for item in tqdm(summary, desc="progress"):
    src, mt_en, ape_en = item["src"], item["mt"], item["ape"]
    back_mt_ko  = back_translate(mt_en)
    back_ape_ko = back_translate(ape_en)
    item.update(back_mt_ko=back_mt_ko, back_ape_ko=back_ape_ko)

    embs = embed_model.encode([src, back_mt_ko, back_ape_ko, ape_en], batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    src_emb, back_mt_emb, back_ape_emb, ape_en_emb = embs
    item["cos_back_mt"] = float(np.dot(src_emb, back_mt_emb))
    item["cos_back_ape"] = float(np.dot(src_emb, back_ape_emb))
    item["cos_direct"] = float(np.dot(src_emb, ape_en_emb))
    item["delta_direct_cos"] = item["cos_direct"] - item["cosine_score"]

    new_com = float(comet.predict([{"src": src, "mt": ape_en}], batch_size=32, progress_bar=False)["scores"][0])
    item["new_comet"]  = new_com
    item["delta_comet"] = new_com - item["comet_score"]

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))
print(f"total of {len(summary)} done")
