# 멋사 QA 챗봇 (AI_NLP_BuffaloWing)

<p align="center">
  <img src="https://github.com/user-attachments/assets/544a3643-8c6d-4b0e-8ded-6dc7a2d0fc82" alt="LikeLion_실전프로젝트02__시연영상_압축본">
</p>

실전 프로젝트 2에서 구축한 **QA 챗봇 + 평가 파이프라인**입니다.  
FastAPI로 모델을 서빙하고, React 프론트에서 채팅하며, RLHF 평가 노트북으로 품질을 분석합니다.

## 핵심 기능
- **FastAPI 서버**: `chatbot/pipeline.py`의 파이프라인을 바로 서빙.
- **Pipeline Adapter**: 파이프라인 인터페이스가 달라도 자동 매핑 호출.
- **React UI**: 채팅 UI + 상태/지연시간 표시.
- **RLHF 평가**: 베이스/정책 모델 비교 + RM/GPT/스타일 지표 분석.
- **ngrok 지원**: 외부 공유 URL로 즉시 테스트.

## 구조
```
chatbot/                # 모델 파이프라인
server/                 # FastAPI 서버 + 어댑터
frontend/               # React UI
evaluation/             # RLHF 평가 노트북
data/                   # 샘플/라벨 데이터
documents/              # 생성 결과, 평가 리포트, 트래킹 이미지
USAGE.md                # 실행 가이드
README.md               # 현재 문서
```

## 빠른 시작 (로컬)

### 1) 서버
```bash
pip install -r server/requirements.txt
python -m server.main
```

환경변수(선택):
- `PIPELINE_MODULE=chatbot.pipeline`
- `MODEL_ID=exaone-4.0-1.2b`  
  (예: Kanana 8B RLHF 모델을 쓴다면 `MODEL_ID=kanana-1.5-8b-sft-merged`)
- `ENABLE_NGROK=1` / `NGROK_AUTHTOKEN=...`

### 2) 프론트
```bash
cd frontend
npm install
npm run dev
```
`frontend/.env`에 API 주소 설정:
```
VITE_API_BASE=http://localhost:8000
```

## API
- `GET /health` → 상태 확인
- `POST /api/chat`
```json
{
  "message": "안녕하세요",
  "history": [
    { "role": "user", "content": "이전 질문" },
    { "role": "assistant", "content": "이전 답변" }
  ]
}
```

## 파이프라인 흐름
1. 사용자 질문 입력
2. 히스토리 + 정책/퓨샷 템플릿 구성
3. `apply_chat_template`로 프롬프트 변환
4. 모델 추론 (`generate`)
5. 프롬프트 에코 제거 + `strip()`
6. (옵션) 실패 시 재시도

## RLHF 평가
- `evaluation/rlhf_eval.ipynb`에서 수행.
- 2단계 구성:
  1) 모델 출력 생성 → JSONL 저장  
  2) RM/GPT/스타일 지표 계산 → 리포트/아티팩트 생성
- 저장 결과:
  - 리포트: `.../evaluation/rlhf_eval_report_*.md`  
  - 생성: `.../evaluation/artifacts_*/generations/*.jsonl`  
  - 지표: `.../evaluation/artifacts_*/eval/*.csv|*.json`

### 평가 요약 (documents/rlhf_evaluation_report.md)
- 데이터: 60개 파일, 샘플 100개 사용
- RM 점수: base mean -0.99 / rlhf mean -1.06 (std 비슷)
- GPT 선호: RLHF 승률 0.95 (A=RLHF, B=Base)
- 스타일 지표(예시, mean ± std):
  - 토큰 수: base 198.4 ±36.1 / rlhf 201.2 ±31.5
  - n-gram 중복(2): base 0.103 / rlhf 0.107
  - 문장 유사도 평균: base 0.029 / rlhf 0.028
  - 질문 키워드 재사용: 1.0 / 1.0
  - assertive 비율: base 0.0033 / rlhf 0.0025
  - risky 키워드: base 0.0012 / rlhf 0.0014

## 생성 예시 (documents/*generations_20260121_161654.jsonl)
- Q: 동물 인식/교육 관심?  
  - Base: "You are a helpful assistant..." 프롬프트 에코 포함, 설명형 응답  
  - RLHF: 유사 구조, 정책 튜닝된 서술  
- Q: 언론이 사회에 미치는 영향?  
  - Base: 사회적 영향 목록화, 일반 설명  
  - RLHF: 구조화된 요약, 영향 요소를 더 또렷하게 제시  
- 파일: `documents/base_generations_20260121_161654.jsonl`, `documents/rlhf_generations_20260121_161654.jsonl`

## 트래킹/시각화 (documents)
- `rlhf_loss_tracking.png`, `rlhf_kl_reward_tracking.png`: 학습 중 loss/kl 추이
- `sft_rouge_score.png`: SFT Rouge 지표
- `rlhf_1st_step_examples.csv`, `rlhf_final_step_examples.csv`: 단계별 예시 비교

## 문제 해결
- **config.json 오류**: `MODEL_ID`가 로컬 경로인데 `config.json`이 없거나 경로가 잘못됨.
- **CORS 에러**: `CORS_ORIGINS`에 프론트 주소 추가.
- **ngrok 미동작**: `ENABLE_NGROK=1`과 `NGROK_AUTHTOKEN` 확인.

## 참고
- 실행 상세는 `USAGE.md`를 확인하세요.
