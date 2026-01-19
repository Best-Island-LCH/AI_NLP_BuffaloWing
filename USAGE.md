# Usage Guide

## Server

Install dependencies:

```bash
pip install -r server/requirements.txt
```

Optional environment setup:
- Copy `server/.env.example` to `server/.env`
- Key options:
  - `PIPELINE_MODULE=chatbot.pipeline`
  - `MODEL_ID=exaone-4.0-1.2b`
  - `ENABLE_NGROK=1`
  - `NGROK_AUTHTOKEN=...`

Run the API:

```bash
python -m server.main
```

Health check:
- `GET http://localhost:8000/health`

Chat endpoint:
- `POST http://localhost:8000/api/chat`
- Body example:
```json
{
  "message": "안녕",
  "history": [
    { "role": "user", "content": "이전 질문" },
    { "role": "assistant", "content": "이전 답변" }
  ]
}
```

## Frontend

Install dependencies:

```bash
cd frontend
npm install
```

Set API base URL:
- Copy `frontend/.env.example` to `frontend/.env`
- Example:
```
VITE_API_BASE=http://localhost:8000
```

Run the app:

```bash
npm run dev
```

Open in browser:
- `http://localhost:5173`

