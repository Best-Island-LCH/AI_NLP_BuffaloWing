import os
import time
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from server.pipeline_adapter import PipelineAdapter

load_dotenv("../.env")

APP_HOST = os.getenv("HOST", "0.0.0.0")
APP_PORT = int(os.getenv("PORT", "8000"))

origins_env = os.getenv("CORS_ORIGINS", "*")
origins = [o.strip() for o in origins_env.split(",") if o.strip()]
if "*" in origins:
    origins = ["*"]

app = FastAPI(title="Chatbot API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

adapter = PipelineAdapter()
_ngrok_tunnel = None


class Message(BaseModel):
    role: str = Field(..., examples=["user", "assistant"])
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Message]] = None


class ChatResponse(BaseModel):
    response: str
    latency_ms: float


@app.on_event("startup")
def _start_ngrok() -> None:
    global _ngrok_tunnel
    if os.getenv("ENABLE_NGROK", "0") != "1":
        return
    try:
        from pyngrok import ngrok

        token = os.getenv("NGROK_AUTHTOKEN")
        if token:
            ngrok.set_auth_token(token)
        _ngrok_tunnel = ngrok.connect(APP_PORT, "http")
        print(f"Ngrok URL: {_ngrok_tunnel.public_url}")
    except Exception as exc:
        print(f"Ngrok disabled: {exc}")


@app.on_event("shutdown")
def _stop_ngrok() -> None:
    global _ngrok_tunnel
    if _ngrok_tunnel is None:
        return
    try:
        from pyngrok import ngrok

        ngrok.disconnect(_ngrok_tunnel.public_url)
        ngrok.kill()
    except Exception:
        pass


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    start = time.perf_counter()
    try:
        history = [m.model_dump() for m in req.history] if req.history else None
        response = adapter.generate(req.message, history=history)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    latency_ms = (time.perf_counter() - start) * 1000.0
    return ChatResponse(response=response, latency_ms=latency_ms)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.main:app", host=APP_HOST, port=APP_PORT, reload=True)
