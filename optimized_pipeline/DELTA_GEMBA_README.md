# Delta GEMBA 계산 가이드

기존 v13 APE 결과에 `delta_gemba` 점수를 추가하는 방법입니다.

## 🎯 목적

- 기존 COS, COMET 점수는 그대로 유지
- APE 개선된 텍스트에 대해서만 GEMBA 점수 재계산
- `delta_gemba = ape_gemba - original_gemba` 추가
- 더 정확한 품질 등급 분류 가능

## 📋 사전 준비

1. **필요한 파일 확인**:
   ```
   optimized_pipeline/
   ├── out/v13/ape_evidence.json  # 기존 APE 결과
   ├── out/v13/gemba.json         # 원본 GEMBA 점수 (참조용)
   ├── add_delta_gemba.py         # 계산 스크립트
   ├── gemba_batch.py             # GEMBA 평가 모듈
   └── cfg.py                     # 설정 파일
   ```

2. **환경 설정**:
   ```bash
   # OpenAI API 키 설정 (GEMBA 평가용)
   export OPENAI_API_KEY="your-api-key"
   ```

## 🚀 실행 방법

### 방법 1: 간단 실행 (권장)
```bash
cd optimized_pipeline
python run_delta_gemba.py
```

### 방법 2: 직접 실행
```bash
cd optimized_pipeline
python add_delta_gemba.py --version v13
```

### 방법 3: 강제 덮어쓰기
```bash
cd optimized_pipeline
python add_delta_gemba.py --version v13 --force
```

## 📊 결과 파일

실행 후 다음 파일이 생성됩니다:

```
out/v13/ape_evidence_with_delta_gemba.json
```

**파일 구조**:
```json
{
  "metadata": {
    "created_at": "2025-01-28T...",
    "source_version": "v13",
    "delta_gemba_added": true,
    "script": "add_delta_gemba.py"
  },
  "statistics": {
    "count": 485,
    "mean": 0.123,
    "improvement_rate": 67.2,
    "positive_count": 326,
    "negative_count": 159
  },
  "records": [
    {
      "key": "example_key",
      "src": "원문",
      "mt": "기계번역",
      "ape": "개선된번역",
      "gemba": 75.0,           // 원본 GEMBA
      "ape_gemba": 82.0,       // APE 후 GEMBA
      "delta_gemba": 7.0,      // 개선도 (새로 추가!)
      "comet": 0.75,
      "delta_comet": 0.05,
      "cos": 0.82,
      "delta_cos": 0.03
    }
  ]
}
```

## 🌐 API 서버 재시작

계산 완료 후 API 서버를 재시작하면 새로운 데이터가 적용됩니다:

```bash
cd api
python main.py
```

API는 자동으로 `ape_evidence_with_delta_gemba.json`을 우선 로드합니다.

## 📈 예상 효과

### 개선 전 (기존)
- APE 개선 후에도 품질 등급이 변하지 않는 경우가 많음
- GEMBA 점수 변화가 반영되지 않음

### 개선 후 (delta_gemba 추가)
- APE 개선 시 GEMBA 점수 변화까지 반영
- 더 정확한 품질 등급 분류
- "나쁨 → 양호" 같은 등급 변화 정확히 표시

## 🔍 통계 예시

```
📊 Delta GEMBA 통계:
- 총 APE 레코드: 485
- 평균 개선도: +2.34
- 개선된 레코드: 326 (67.2%)
- 악화된 레코드: 159
- 변화없음: 0
- 최대 개선: +15.0
- 최대 악화: -8.0
```

## ⚠️ 주의사항

1. **API 비용**: GEMBA 평가는 OpenAI API를 사용하므로 비용이 발생합니다
2. **처리 시간**: APE 레코드 수에 따라 몇 분에서 십여 분 소요
3. **배치 크기**: `cfg.py`의 `GEMBA_BATCH` 설정으로 조절 가능

## 🛠️ 문제 해결

### 파일을 찾을 수 없음
```bash
❌ APE evidence file not found: out/v13/ape_evidence.json
```
→ `optimized_pipeline` 디렉토리에서 실행했는지 확인

### API 키 오류
```bash
❌ OpenAI API key not found
```
→ `.env` 파일에 `OPENAI_API_KEY` 설정 확인

### 메모리 부족
→ `cfg.py`에서 `GEMBA_BATCH` 크기를 줄이기 (기본값: 4)

## 📝 추가 옵션

- `--version`: 처리할 버전 지정 (기본값: v13)
- `--force`: 기존 출력 파일 강제 덮어쓰기
