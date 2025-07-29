import asyncio
from gemba_batch import gemba_batch

async def test_gemba():
    # 실제 파이프라인에서 0점을 받은 케이스 테스트
    src_texts = [
        "라인 1: 매크로를 함수처럼 사용했기 때문에 문제가 발생했습니다.\n라인 1: 매크로에 전체 괄호가 누락되었습니다.\n라인 4: 매크로가 적용되면 수식이 (((24+(a))*(a))*(a)) 와 같이 계산됩니다. 의도하지 않은 연산이기 때문에 문제가 발생했습니다."
    ]
    target_texts = [
        "Line 1: A macro is used like a function.  \nLine 1: A macro is not enclosed with parentheses.  \nLine 4: If the macro is applied, then this line is evaluated into (((24+(a))*(a))*(a)). This will produce an unexpected output."
    ]
    
    print("Testing GEMBA with actual pipeline data...")
    print(f"Source: {src_texts[0]}")
    print(f"Target: {target_texts[0]}")
    
    results = await gemba_batch(src_texts, target_texts)
    print(f"Results: {results}")
    
    for i, result in enumerate(results):
        print(f"Result {i}: {result}")
        if isinstance(result, tuple):
            overall, adequacy, fluency, evidence = result
            print(f"  Overall: {overall}")
            print(f"  Adequacy: {adequacy}")
            print(f"  Fluency: {fluency}")
            print(f"  Evidence: {evidence}")

if __name__ == "__main__":
    asyncio.run(test_gemba())
