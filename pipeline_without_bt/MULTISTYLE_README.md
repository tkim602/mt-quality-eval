# Multi-Style MT Evaluation Pipeline

This directory contains a multi-style version of the MT evaluation pipeline that can handle 5 different text styles with appropriate evaluation criteria and post-editing approaches.

## Overview

The multi-style pipeline extends the original technical-focused pipeline to handle diverse text types:

1. **Formal/Technical** - Original functionality for technical documentation, business communications
2. **News/Journalism** - News articles, press releases, reports with factual accuracy focus
3. **Casual/Conversation** - Informal chat, social media, everyday communication
4. **Literature** - Novels, short stories, essays with artistic merit preservation
5. **Poetry** - Poems, verses, song lyrics with rhythm and beauty focus

## Key Features

### Automatic Style Detection
- Uses pattern matching to automatically detect text style from Korean source
- Fallback to 'formal' style for unclassified text
- Manual style override supported

### Style-Specific Evaluation (GEMBA)
- **Formal**: Technical terminology consistency, professional tone
- **News**: Factual accuracy, journalistic clarity, objectivity  
- **Casual**: Natural conversation flow, emotional tone, informal expressions
- **Literature**: Artistic merit, metaphor preservation, cultural adaptation
- **Poetry**: Rhythm and flow, imagery, emotional resonance

### Style-Specific Post-Editing (APE)
- **Formal**: Professional language, technical precision
- **News**: Clear reporting style, factual integrity
- **Casual**: Authentic informal English (ㅋㅋㅋ→lol, 대박→awesome)
- **Literature**: Sophisticated prose, literary atmosphere
- **Poetry**: Poetic beauty, rhythmic quality

## Files

### Core Multi-Style Components
- `gemba_batch_multistyle.py` - Multi-style GEMBA evaluation
- `ape_multistyle.py` - Multi-style APE post-editing  
- `run_pipeline_multistyle.py` - Complete pipeline orchestrator

### Original Components (Still Used)
- `filter.py` - Initial filtering and validation
- `cfg.py` - Configuration settings
- Other utility files

## Usage

### Quick Start
```bash
# Run complete multi-style pipeline
python run_pipeline_multistyle.py

# Run to specific directory
python run_pipeline_multistyle.py /path/to/output/dir
```

### Individual Components
```bash
# Multi-style GEMBA only
python gemba_batch_multistyle.py

# Multi-style APE only  
python ape_multistyle.py
```

## Configuration

The multi-style pipeline uses the same `cfg.py` settings as the original pipeline:

- `GEMBA_MODEL` - OpenAI model for evaluation
- `APE_MODEL` - OpenAI model for post-editing
- `GEMBA_BATCH` - Batch size for evaluation
- `APE_CONCURRENCY` - Concurrency for post-editing

## Style Detection Patterns

The system uses regex patterns to detect text styles:

### Formal/Technical
- Formal verb endings: `합니다|습니다|됩니다|입니다`
- Technical terms: `사용자|시스템|데이터베이스|오류|서버`
- Formal expressions: `에 대한|에 관한|관련하여|따라서`

### News/Journalism  
- News terminology: `기자|뉴스|보도|발표했다|전했다|밝혔다`
- Authority figures: `정부|대통령|국회|장관|시장`
- Time expressions: `오늘|어제|내일|지난|다음|이번`

### Casual/Conversation
- Internet expressions: `ㅋㅋ|ㅎㅎ|ㅠㅠ|헉|대박|레알`
- Informal pronouns: `야|너|나|우리|얘|걔|쟤`
- Casual endings: `했어|했네|했지|거야|거지|겠어`

### Literature
- Literary themes: `마음|영혼|운명|사랑|아름다운|슬픈`
- Nature imagery: `하늘|바다|달|별|꽃|나무|새|바람`
- Literary endings: `였다|였네|였구나|하였다|하더라`

### Poetry
- Poetic themes: `님|그대|사랑|그리움|눈물|한숨|꿈`
- Natural imagery: `바람|꽃|달|별|하늘|구름|새|물`
- Poetic endings: `하네|하노|하구나|로다|이다|이로다`

## Output Structure

Multi-style pipeline output includes additional fields:

```json
{
  "key": "item_id",
  "src": "Korean source text",
  "mt": "Original machine translation", 
  "ape": "Style-appropriate post-edited text",
  "detected_style": "formal|news|casual|literature|poetry",
  "ape_style": "Style used for APE",
  "gemba": 85.0,
  "ape_gemba": 92.0,
  "delta_gemba": 7.0,
  "tag": "soft_pass|fail|strict_pass"
}
```

## Style-Specific Examples

### Formal/Technical
```
Korean: "사용자 인증이 실패했습니다."
MT: "User authentication failed."  
APE: "User authentication failed." (minimal change needed)
```

### Casual/Conversation
```
Korean: "그래서 그런거였어?"
MT: "That is what it happened"
APE: "Oh, so that's what it was?"
```

### News/Journalism
```
Korean: "대통령이 오늘 새로운 정책을 발표했습니다."
MT: "The President announced a new policy today."
APE: "The President announced a new policy today." (already good)
```

### Literature  
```
Korean: "그의 마음속에는 깊은 슬픔이 자리하고 있었다."
MT: "Deep sorrow was placed in his heart."
APE: "Deep sorrow had settled in his heart."
```

### Poetry
```
Korean: "바람이 불어와 / 내 마음을 흔들어 놓고 간다"
MT: "The wind blows / Shaking my heart and goes"  
APE: "The wind comes blowing / Shaking my heart and passing by"
```

## Statistics and Analysis

The pipeline provides detailed statistics:

### Style Distribution
Shows how many items were detected for each style:
```
    FORMAL: 1250 items (62.5%) - Avg GEMBA: 82.3
      NEWS:  300 items (15.0%) - Avg GEMBA: 78.9
    CASUAL:  250 items (12.5%) - Avg GEMBA: 75.4
LITERATURE:  150 items ( 7.5%) - Avg GEMBA: 79.8
    POETRY:   50 items ( 2.5%) - Avg GEMBA: 81.2
```

### APE Improvement Summary
Shows improvement/degradation counts and average deltas:
```
   COS:   45 improved,   12 degraded,    8 unchanged
         Average delta: +0.023

 COMET:   52 improved,    9 degraded,    4 unchanged  
         Average delta: +0.087

 GEMBA:   48 improved,   11 degraded,    6 unchanged
         Average delta: +4.215
```

## Benefits of Multi-Style Approach

1. **Appropriate Evaluation**: Each text style evaluated with relevant criteria
2. **Better Post-Editing**: Style-aware APE produces more natural results
3. **Reduced Over-Correction**: Casual expressions no longer "corrected" to formal language
4. **Cultural Sensitivity**: Better handling of Korean cultural expressions
5. **Domain Flexibility**: Single pipeline handles diverse content types

## Comparison with Original Pipeline

| Aspect | Original Pipeline | Multi-Style Pipeline |
|--------|-------------------|---------------------|
| Text Types | Technical/Formal only | 5 diverse styles |
| Evaluation | Technical criteria | Style-appropriate criteria |
| Post-Editing | Formal improvements | Style-aware improvements |
| Casual Text | Over-corrects | Natural handling |
| Cultural Expressions | May lose nuance | Preserves style-appropriate expressions |
| Flexibility | Single domain | Multi-domain |

## Future Enhancements

1. **Additional Styles**: Academic, legal, medical text styles
2. **Style Mixing**: Handle texts with multiple styles
3. **Custom Patterns**: User-defined style detection patterns
4. **Style Confidence**: Confidence scores for style detection
5. **Specialized Models**: Style-specific fine-tuned models
