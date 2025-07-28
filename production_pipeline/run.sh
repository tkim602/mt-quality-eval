#!/bin/bash
# Production Pipeline 실행 스크립트

echo "🚀 MT Quality Evaluation Pipeline (Production) 🚀"
echo ""

# 1. 파이프라인 실행
echo "📊 1단계: 파이프라인 실행 중..."
python run_pipeline.py

if [ $? -ne 0 ]; then
    echo "❌ 파이프라인 실행 실패"
    exit 1
fi

echo "✅ 파이프라인 완료"
echo ""

# 2. API 서버 실행 여부 확인
read -p "🌐 API 서버를 실행하시겠습니까? (y/n): " start_api

if [ "$start_api" = "y" ] || [ "$start_api" = "Y" ]; then
    echo "🌐 2단계: API 서버 실행 중..."
    cd api
    echo "웹 인터페이스: http://localhost:8000"
    python main.py
else
    echo "📁 결과는 data/out/ 폴더에서 확인할 수 있습니다"
fi

echo ""
echo "🎉 완료!"
