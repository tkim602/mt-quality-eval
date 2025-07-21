import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import precision_score, recall_score

JSON_PATH = Path("sample_tagged_human_merged.json")
df = pd.read_json(JSON_PATH)

df["label"] = df["manual"].map({"Pass": 1, "Fail": 0}).astype(int)

if "cos" not in df.columns or "comet" not in df.columns:
    print("computing comet, cos sim")
    from sentence_transformers import SentenceTransformer, util
    from comet import download_model, load_from_checkpoint
    from tqdm import tqdm

    labse = SentenceTransformer("sentence-transformers/LaBSE")
    src_emb = labse.encode(df["src"].tolist(), batch_size=128, normalize_embeddings=True)
    mt_emb  = labse.encode(df["mt"].tolist(),  batch_size=128, normalize_embeddings=True)
    df["cos"] = (src_emb * mt_emb).sum(axis=1)

    ckpt = download_model("Unbabel/wmt22-cometkiwi-da")
    comet_mdl = load_from_checkpoint(ckpt)
    df["comet"] = comet_mdl.predict(
        [{"src": s, "mt": m} for s, m in zip(df["src"], df["mt"])],
        batch_size=64, progress_bar=True
    )["scores"]

    JSON_PATH.write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    print(f"saved")

# ── 2. 탐색 범위 & 목표 설정 ───────────────────────
cos_grid   = np.arange(0.74, 0.59, -0.01)
comet_grid = np.arange(0.86, 0.77, -0.01)

PREC_MIN = 0.95     # precision 하한
REC_MIN  = 0.10     # result 표시 최소 recall

cands = []          # 후보 저장 (cos, comet, prec, rec, support)

# ── 3. 전체 그리드 탐색 ──────────────────────
for c_thr in cos_grid:
    for m_thr in comet_grid:
        pred = (df["cos"] >= c_thr) & (df["comet"] >= m_thr)
        supp = int(pred.sum())
        if supp == 0:
            continue

        p = precision_score(df["label"], pred, zero_division=0)
        if p < PREC_MIN:
            continue

        r = recall_score(df["label"], pred, zero_division=0)
        if r < REC_MIN:
            continue

        cands.append((c_thr, m_thr, p, r, supp))

# ── 4. recall → precision → support 순 정렬 ──
cands.sort(key=lambda x: (-x[3], -x[2], -x[4]))

# ── 5. 결과 출력 (상위 20개) ──────────────────
header = f" precision≥{PREC_MIN},  후보 {len(cands)}개\n"
print(header + "-"*55)
print(" rank |  cos  | comet |  precision |  recall | support")
print("-"*55)
for i, (c_thr, m_thr, p, r, supp) in enumerate(cands[:20], 1):
    print(f"{i:>4} | {c_thr:0.3f} | {m_thr:0.3f} |   {p:0.3f}    |  {r:0.3f} | {supp:>5}")

# 최적 컷 저장
if cands:
    best = cands[0]
    print(f"\n★ 최적 후보: cos≥{best[0]:.3f}, comet≥{best[1]:.3f}  "
          f"(precision={best[2]:.3f}, recall={best[3]:.3f}, support={best[4]})")
else:
    print("\n조건을 만족하는 조합이 없습니다.")