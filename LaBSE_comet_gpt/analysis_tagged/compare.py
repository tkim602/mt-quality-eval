import json
import pandas as pd

# JSON 파일 경로를 알맞게 설정하세요
json_path = '../eval_tagged/out/ape_eval.json'

# 1) 데이터 로드
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
df = pd.DataFrame(data)

# 2) 그룹별 cos_direct & new_comet 평균 계산
means_abs = df.groupby('group')[['cosine_score', 'comet_score', 'cos_direct', 'new_comet', 'delta_direct_cos', 'delta_comet']].mean()

# 3) 결과 출력
print(means_abs)
