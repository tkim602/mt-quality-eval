from sentence_transformers import SentenceTransformer
import numpy as np
import time
from transformers import AutoTokenizer

import os
import openai
from dotenv import load_dotenv



load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_EMB = os.getenv("openai", "text-embedding-3-large")


def get_openai_embedding(text, model=MODEL_EMB):
    response = openai.Embedding.create(
        input=text,
        model=model,
        api_key=OPENAI_API_KEY
    )
    return np.array(response.data[0].embedding)

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

models = ["nlpai-lab/KoE5", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
          "google-bert/bert-base-multilingual-uncased", "BM-K/KoSimCSE-roberta",
          "Alibaba-NLP/gte-multilingual-base", "intfloat/multilingual-e5-large", "nlpai-lab/KURE-v1", "upskyy/bge-m3-korean", "upskyy/e5-large-korean", "upskyy/e5-small-korean",
          "sentence-transformers/LaBSE", "beademiguelperez/sentence-transformers-multilingual-e5-small", "bespin-global/klue-sroberta-base-continue-learning-by-mnr", "dragonkue/BGE-m3-ko"]

# 1개씩 테스팅
# models = ["nlpai-lab/KURE-v1"]

true_pairs = [("The rear cover 130a is disposed and coupled to the middle cover 120a and protects the middle cover 120a.", "후면 커버(130a)는 미들 커버(120a) 상에 배치되어 결합되고, 미들 커버(120a)를 보호한다"),
              ("In an configuration of the present invention, the layout of the panel and the cutting line may include non-linear lines.", "본 발명의 실시예에 있어서, 상기 패널의 레이아웃 및 상기 재단라인은 비선형 라인을 포함할 수 있다."),
              ("In this case, the video is decoded through the first media restoring unit 143 implemented in JavaScript (S2300).", "이 때는 자바스크립트로 구현된 제1 미디어 복원부(143)를 통해 비디오를 디코딩한다(S2300)."),
              ("From a security point of view, it may be advantageous to place critical security components outside the protected VM.", "보안 관점에서는 보호되는 VM 외부에 중요한 보안 구성 요소를 위치시키는 것이 유리할 수 있다."),
              ("Meanwhile, application services can be incarnated by using the user environment interface provided by ODC and the Python language.", "한편 애플리케이션 응용 서비스는 ODC에서 제공하는 사용자 환경 인터페이스와 파이썬 언어를 사용하여 구현할 수 있다.")]

false_pairs = [("FIG. 10 shows views of a display optimized for a user according to embodiments of the present invention.", "다목적실용위성 자료의 오픈 데이터 큐브 적용을 위한 기본 고려사항"),
               ("The command criterion for a predetermined command in the voice input may be based on whether some keywords are included in the voice input, for example", "하나 이상의 실시예들에 의하면, 조화기는 공간 벽에 대항하여 장착되고 재생기는 빌딩 실외에 장착된다."),
               ("The first terminal may determine a second control information transmission start time point based on the information.", "GMDN은 대분류에 속하지 않는 새로운 의료기기의 출현이나 향후 대분류의 변경에 대비하여 명칭이 할당되어지지 않은 4개의 대분류를 포함하고 있다."),
               ("The electronic device generates a vibration pattern based on the vibration parameter in operation S505.", "급성기 뇌경색 환자에서 인지 및 연하 기능과 일상생활 독립성"),
               ("In order to examine the change of pores in more detail, the distribution of pores was obtained through the HK method, and it is shown in Figure 4.", "단계 511에서, 캐시 서비스는 캐시에 저장된 컨텐트를 미디어 플레이어에게 스트리밍한다.")] 


for name in models:
    # tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    print(f"\nModel: {name}")
    for label, pairs in [("True", true_pairs), ("False", false_pairs)]:
        model = SentenceTransformer(name, trust_remote_code=True)
        scores = []              
        for en, ko in pairs:
                # if name == "openai":
                #     emb_en = get_openai_embedding(en)
                #     emb_ko = get_openai_embedding(ko)
                # else:
                emb_en = model.encode(en, convert_to_numpy=True)
                emb_ko = model.encode(ko, convert_to_numpy=True)
                score = cos_sim(emb_en, emb_ko)
                scores.append(score)
            # print(tokenizer.tokenize(ko))
        mean = np.mean(scores)
        print(f"{label}: scores: {[f'{s}' for s in scores]}, mean: {mean}")