import pandas as pd
import matplotlib.pyplot as plt

# 1) validation.json 불러오기 (경로는 실제 파일 위치에 맞게 조정)
df = pd.read_json("../out/validation.json")

# 2) 버킷 순서 지정
bucket_order = ["short", "medium", "long", "very_long"]
df["bucket"] = pd.Categorical(df["bucket"], categories=bucket_order, ordered=True)

# 3) 그릴 지표 리스트
metrics = ["cos", "comet", "gemba", "human_mqm"]

for m in metrics:
    # 각 버킷별로 값 추출
    data = [df[df["bucket"] == b][m].dropna().values for b in bucket_order]
    
    plt.figure()
    # 박스플롯, showmeans=True 로 평균 표시
    plt.boxplot(
        data,
        labels=bucket_order,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markeredgecolor": "black",
            "markerfacecolor": "firebrick"
        },
        flierprops={"marker": "o", "markerfacecolor": "gray", "alpha": 0.6}
    )
    plt.title(f"Distribution of {m} by Bucket\n(mean=♦, outliers=○)")
    plt.xlabel("Bucket")
    plt.ylabel(m)
    plt.tight_layout()
    plt.show()
