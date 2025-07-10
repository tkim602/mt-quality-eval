#!/usr/bin/env python3
from pathlib import Path
import os, orjson, numpy as np, openai
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from comet import download_model, load_from_checkpoint


from dotenv import load_dotenv
load_dotenv()

APE_JSON = Path("out/ape_improved_prompt.json")
OUT_JSON = Path("out/ape_eval.json")

openai.api_key = os.getenv("OPENAI_API_KEY")
embed_model = SentenceTransformer("sentence-transformers/LaBSE")
comet = load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da"))

summary = orjson.loads(APE_JSON.read_bytes())
for item in tqdm(summary, desc="progress"):
    src = item["src"]             
    orig_cos = item["cosine_score"]     
    orig_comet = item["comet_score"]     
    mt_en = item["mt"]       
    ape_en = item["ape"]          

    # 원본영어 -> ko2 백번역
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system",
             "content":
                "Translate the following text from English to Korean. This is to verfify the accuracy of the inital Korean to English translation."},
            {"role":"user",  
             "content": mt_en}
        ],
        temperature=0
    )
    back_mt_ko = response.choices[0].message.content.strip()
    item["back_mt_ko"] = back_mt_ko

    # ape → ko3 백번역
    resp_ape = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system",
             "content":""
                "Translate the following text from English to Korean. This is to verfify the accuracy of the inital Korean to English translation."},
            {"role":"user",
             "content": ape_en}
        ],
        temperature=0
    )
    back_ape_ko = resp_ape.choices[0].message.content.strip()
    item["back_ape_ko"] = back_ape_ko

    # ko–ko2 코사인 비교
    embs_back_mt = embed_model.encode([src, back_mt_ko], batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    cos_back_mt = float(np.dot(embs_back_mt[0], embs_back_mt[1]))
    item["cos_back_mt"] = cos_back_mt

    # ko-ko3 비교 
    embs_back_ape = embed_model.encode([src, back_ape_ko], batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    cos_back_ape = float(np.dot(embs_back_ape[0], embs_back_ape[1]))
    item["cos_back_ape"] = cos_back_ape

    # en–ko direct 비교 (원문-APE)
    embs_direct = embed_model.encode([src, ape_en], batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    cos_direct = float(np.dot(embs_direct[0], embs_direct[1]))
    item["cos_direct"] = cos_direct
    item["delta_direct_cos"] = cos_direct - orig_cos

    # new comet 
    new_com = float(comet.predict([{"src": src, "mt": ape_en}], batch_size=32, progress_bar=False)["scores"][0])
    item["new_comet"] = new_com
    item["delta_comet"] = new_com - orig_comet

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS))

print(f"total of {len(summary)} done")
