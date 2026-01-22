# RLHF Evaluation Report

## Dataset
- Source files: 60
- Samples used: 100

## RM Score
- base: mean=-0.9929, std=3.2960
- rlhf: mean=-1.0575, std=3.3034

## GPT Preference (A=RLHF, B=Base)
- RLHF win rate: 0.950

## Style Metrics (mean ± std)
### token_count
- base: 198.4100 ± 36.0628
- rlhf: 201.1500 ± 31.4573

### paragraph_count
- base: 12.0400 ± 2.8138
- rlhf: 12.1400 ± 2.5938

### avg_sentences_per_paragraph
- base: 2.5816 ± 0.6197
- rlhf: 2.5778 ± 0.4139

### list_usage_ratio
- base: 0.5246 ± 0.1980
- rlhf: 0.5362 ± 0.1528

### intro_body_conclusion
- base: 0.0000 ± 0.0000
- rlhf: 0.0000 ± 0.0000

### first_sentence_length
- base: 1.0000 ± 0.0000
- rlhf: 1.0000 ± 0.0000

### last_sentence_summary
- base: 0.0200 ± 0.1407
- rlhf: 0.0100 ± 0.1000

### ngram_rep_2
- base: 0.1027 ± 0.0595
- rlhf: 0.1065 ± 0.0610

### ngram_rep_3
- base: 0.0346 ± 0.0362
- rlhf: 0.0378 ± 0.0372

### ngram_rep_4
- base: 0.0170 ± 0.0252
- rlhf: 0.0181 ± 0.0252

### sentence_similarity_avg
- base: 0.0290 ± 0.0168
- rlhf: 0.0280 ± 0.0129

### phrase_repeat_rate
- base: 0.0111 ± 0.0158
- rlhf: 0.0099 ± 0.0151

### keyword_overuse_ratio
- base: 0.1334 ± 0.0317
- rlhf: 0.1342 ± 0.0317

### formal_ending_ratio
- base: 0.0004 ± 0.0036
- rlhf: 0.0004 ± 0.0036

### colloquial_ratio
- base: 0.0017 ± 0.0112
- rlhf: 0.0019 ± 0.0092

### exclaim_ratio
- base: 0.0167 ± 0.0427
- rlhf: 0.0123 ± 0.0304

### question_mark_ratio
- base: 0.0453 ± 0.0543
- rlhf: 0.0402 ± 0.0405

### first_person_ratio
- base: 0.0203 ± 0.0497
- rlhf: 0.0111 ± 0.0269

### imperative_ratio
- base: 0.0354 ± 0.0954
- rlhf: 0.0341 ± 0.0897

### speculative_ratio
- base: 0.0289 ± 0.0451
- rlhf: 0.0239 ± 0.0359

### apology_ratio
- base: 0.0000 ± 0.0000
- rlhf: 0.0000 ± 0.0000

### defensive_ratio
- base: 0.0000 ± 0.0000
- rlhf: 0.0000 ± 0.0000

### hedge_ratio
- base: 0.0012 ± 0.0069
- rlhf: 0.0017 ± 0.0078

### avg_sentence_length
- base: 6.7661 ± 1.6636
- rlhf: 6.7127 ± 1.3972

### subordinate_ratio
- base: 0.0263 ± 0.0363
- rlhf: 0.0175 ± 0.0291

### comma_density
- base: 0.6031 ± 0.2991
- rlhf: 0.5778 ± 0.2702

### conjunction_density
- base: 0.0159 ± 0.0324
- rlhf: 0.0134 ± 0.0245

### fragment_ratio
- base: 0.9979 ± 0.0117
- rlhf: 0.9977 ± 0.0098

### entity_clarity_ratio
- base: 0.4513 ± 0.1219
- rlhf: 0.4427 ± 0.1179

### proper_noun_ratio
- base: 0.0631 ± 0.0474
- rlhf: 0.0627 ± 0.0449

### keyword_coverage
- base: 1.0000 ± 0.0000
- rlhf: 1.0000 ± 0.0000

### info_units_per_sentence
- base: 6.2898 ± 1.5141
- rlhf: 6.2394 ± 1.2437

### stopword_ratio
- base: 0.0514 ± 0.0205
- rlhf: 0.0486 ± 0.0209

### redundant_sentence_ratio
- base: 0.0045 ± 0.0067
- rlhf: 0.0039 ± 0.0038

### assertive_ratio
- base: 0.0033 ± 0.0108
- rlhf: 0.0025 ± 0.0115

### risky_keyword_ratio
- base: 0.0012 ± 0.0086
- rlhf: 0.0014 ± 0.0143

### conditional_ratio
- base: 0.0432 ± 0.0579
- rlhf: 0.0508 ± 0.0830

### neutral_vocab_ratio
- base: 0.0081 ± 0.0200
- rlhf: 0.0112 ± 0.0269

### question_keyword_reuse
- base: 1.0000 ± 0.0000
- rlhf: 1.0000 ± 0.0000

### viewpoint_changes
- base: 0.0100 ± 0.1000
- rlhf: 0.0200 ± 0.1407

### logic_connector_consistency
- base: 0.0159 ± 0.0324
- rlhf: 0.0134 ± 0.0245

### order_stability
- base: 0.8900 ± 0.3145
- rlhf: 0.9200 ± 0.2727

## Notes
- Style metrics use lightweight heuristics. Replace with a richer tokenizer or domain-specific analyzer if needed.
