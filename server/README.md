# Server

Run the API with FastAPI:

```bash
pip install -r server/requirements.txt
python -m server.main
```

Optional ngrok:
```bash
set ENABLE_NGROK=1
set NGROK_AUTHTOKEN=your_token
python -m server.main
```

Health check:
- `GET /health`

Chat endpoint:
- `POST /api/chat`
  - body: `{ "message": "...", "history": [{"role":"user","content":"..."}] }`
