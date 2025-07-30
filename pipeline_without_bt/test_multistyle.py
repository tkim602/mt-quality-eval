# test_multistyle.py - Test Multi-Style Detection and Evaluation

import asyncio
import logging
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from gemba_batch_multistyle import detect_text_style, gemba_batch_multistyle
from ape_multistyle import edit_sentence_multistyle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test samples for each style
TEST_SAMPLES = {
    'formal': [
        "사용자 인증이 실패했습니다.",
        "데이터베이스 연결 오류가 발생했습니다.", 
        "시스템이 정상적으로 종료되었습니다.",
        "보안 설정에 관한 문서를 참조하시기 바랍니다."
    ],
    'news': [
        "대통령이 오늘 새로운 정책을 발표했습니다.",
        "경제 성장률이 지난해 3.2%를 기록했다고 정부가 밝혔다.",
        "이번 조사 결과에 따르면 국민 만족도가 상승했다.",
        "기자회견에서 장관은 향후 계획을 설명했다."
    ],
    'casual': [
        "그래서 그런거였어?",
        "진짜 대박이야! ㅋㅋㅋ",
        "완전 헐~~ 이거 뭐야ㅋㅋ",
        "아 진짜? 나도 그렇게 생각했어!"
    ],
    'literature': [
        "그의 마음속에는 깊은 슬픔이 자리하고 있었다.",
        "달빛이 창문을 통해 스며들어 방 안을 은은하게 밝혔다.",
        "사랑은 때로는 아름답고 때로는 아픈 것이었다.",
        "운명이라는 것이 과연 존재하는 것일까?"
    ],
    'poetry': [
        "바람이 불어와 내 마음을 흔들어 놓고 간다",
        "꽃잎이 떨어져도 향기는 남아있네",
        "그대여, 이 마음을 어찌 전할까",
        "달빛 아래 홀로 서서 그리움에 잠기네"
    ]
}

# Corresponding English translations (simulating MT output)
TEST_TRANSLATIONS = {
    'formal': [
        "User authentication failed.",
        "Database connection error was occurred.",
        "The system was shutdown normally.",
        "Please refer to the document about security settings."
    ],
    'news': [
        "The President announced a new policy today.",
        "The government revealed that last year's economic growth rate recorded 3.2%.",
        "According to this survey result, citizen satisfaction rose.",
        "At the press conference, the minister explained future plans."
    ],
    'casual': [
        "That is what it happened",
        "Really amazing! lol",
        "Totally what~~ what is this lol",
        "Oh really? I thought that too!"
    ],
    'literature': [
        "Deep sorrow was placed in his heart.",
        "Moonlight penetrated through the window and dimly lit the room.",
        "Love was sometimes beautiful and sometimes painful thing.",
        "Does fate really exist?"
    ],
    'poetry': [
        "The wind blows and shakes my heart and goes",
        "Even when petals fall the fragrance remains",
        "My dear, how can I tell this heart",
        "Standing alone under moonlight, immersed in longing"
    ]
}

def test_style_detection():
    """Test automatic style detection"""
    logger.info("Testing style detection...")
    
    for expected_style, samples in TEST_SAMPLES.items():
        logger.info(f"\nTesting {expected_style.upper()} style:")
        
        for sample in samples:
            detected = detect_text_style(sample)
            status = "✓" if detected == expected_style else "✗"
            logger.info(f"  {status} '{sample[:50]}...' -> {detected}")

async def test_multistyle_evaluation():
    """Test multi-style GEMBA evaluation"""
    logger.info("\nTesting multi-style GEMBA evaluation...")
    
    for style in TEST_SAMPLES.keys():
        logger.info(f"\nEvaluating {style.upper()} samples:")
        
        src_texts = TEST_SAMPLES[style]
        tgt_texts = TEST_TRANSLATIONS[style]
        
        try:
            # Test with explicit style
            results = await gemba_batch_multistyle(src_texts, tgt_texts, style)
            
            for i, (src, tgt, result) in enumerate(zip(src_texts, tgt_texts, results)):
                if isinstance(result, tuple):
                    overall, adequacy, fluency, evidence = result
                    logger.info(f"  Sample {i+1}: Overall={overall:.1f}, Adequacy={adequacy:.1f}, Fluency={fluency:.1f}")
                    logger.info(f"    SRC: {src}")
                    logger.info(f"    TGT: {tgt}")
                    logger.info(f"    Evidence: {evidence}")
                else:
                    logger.warning(f"  Sample {i+1}: Unexpected result format: {result}")
                
        except Exception as e:
            logger.error(f"Error evaluating {style} samples: {e}")

async def test_multistyle_ape():
    """Test multi-style APE editing"""
    logger.info("\nTesting multi-style APE editing...")
    
    # Test cases with known issues
    test_cases = [
        ("formal", "데이터베이스 연결 오류가 발생했습니다.", "Database connection error was occurred.", "fail", "문법 오류: 'was occurred' 사용"),
        ("casual", "그래서 그런거였어?", "That is what it happened", "fail", "의미가 불분명하고 문법이 어색함"),
        ("news", "경제 성장률이 3.2%를 기록했다.", "Economic growth rate recorded 3.2%.", "soft_pass", "소소한 개선 필요"),
        ("literature", "그의 마음속에는 깊은 슬픔이 자리하고 있었다.", "Deep sorrow was placed in his heart.", "soft_pass", "문학적 표현 개선 가능"),
        ("poetry", "바람이 불어와 내 마음을 흔들어 놓고 간다", "The wind blows and shakes my heart and goes", "fail", "시적 리듬감 부족")
    ]
    
    for style, src, mt, mode, evidence in test_cases:
        logger.info(f"\nTesting {style.upper()} APE:")
        logger.info(f"  SRC: {src}")
        logger.info(f"  MT:  {mt}")
        logger.info(f"  Mode: {mode}, Evidence: {evidence}")
        
        try:
            ape_result = await edit_sentence_multistyle(src, mt, mode, evidence, style)
            logger.info(f"  APE: {ape_result}")
            
            # Compare with original
            if ape_result != mt:
                logger.info(f"  ✓ APE made changes")
            else:
                logger.info(f"  - APE made no changes")
                
        except Exception as e:
            logger.error(f"  ✗ APE failed: {e}")

def test_auto_detection():
    """Test automatic style detection with mixed samples"""
    logger.info("\nTesting automatic style detection...")
    
    mixed_samples = [
        "사용자 인증이 실패했습니다.",  # formal
        "진짜 대박이야! ㅋㅋㅋ",           # casual
        "대통령이 새로운 정책을 발표했다.",   # news
        "그의 마음속에는 깊은 슬픔이...",    # literature
        "바람이 불어와 내 마음을...",       # poetry
    ]
    
    expected = ['formal', 'casual', 'news', 'literature', 'poetry']
    
    for sample, expected_style in zip(mixed_samples, expected):
        detected = detect_text_style(sample)
        status = "✓" if detected == expected_style else "✗"
        logger.info(f"  {status} '{sample}' -> {detected} (expected: {expected_style})")

async def main():
    """Run all tests"""
    logger.info("="*60)
    logger.info("MULTI-STYLE PIPELINE TESTING")
    logger.info("="*60)
    
    # Test 1: Style detection
    test_style_detection()
    
    # Test 2: Auto detection
    test_auto_detection()
    
    # Test 3: Multi-style evaluation (requires OpenAI API)
    try:
        await test_multistyle_evaluation()
    except Exception as e:
        logger.warning(f"Skipping GEMBA evaluation test (API issue?): {e}")
    
    # Test 4: Multi-style APE (requires OpenAI API)
    try:
        await test_multistyle_ape()
    except Exception as e:
        logger.warning(f"Skipping APE test (API issue?): {e}")
    
    logger.info("\n" + "="*60)
    logger.info("TESTING COMPLETED")
    logger.info("="*60)

if __name__ == "__main__":
    asyncio.run(main())
