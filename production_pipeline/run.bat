@echo off
REM Production Pipeline 실행 스크립트 (Windows)

echo 🚀 MT Quality Evaluation Pipeline (Production) 🚀
echo.

REM 1. 파이프라인 실행
echo 📊 1단계: 파이프라인 실행 중...
python run_pipeline.py

if %errorlevel% neq 0 (
    echo ❌ 파이프라인 실행 실패
    pause
    exit /b 1
)

echo ✅ 파이프라인 완료
echo.

REM 2. API 서버 실행 여부 확인
set /p start_api="🌐 API 서버를 실행하시겠습니까? (y/n): "

if /i "%start_api%"=="y" (
    echo 🌐 2단계: API 서버 실행 중...
    cd api
    echo 웹 인터페이스: http://localhost:8000
    python main.py
) else (
    echo 📁 결과는 data\out\ 폴더에서 확인할 수 있습니다
)

echo.
echo 🎉 완료!
pause
