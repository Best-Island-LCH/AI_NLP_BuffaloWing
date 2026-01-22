import { useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "https://nontheatrical-judiciarily-susanne.ngrok-free.dev";

const starter = [
  {
    role: "assistant",
    content:
      "안녕하세요. Kanana 8B RLHF 모델로 답변합니다. 무엇이든 물어보세요.",
  },
];

export default function App() {
  const [messages, setMessages] = useState(starter);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [latency, setLatency] = useState(null);
  const [error, setError] = useState(null);

  const canSend = input.trim().length > 0 && !loading;
  const messageCount = messages.length;

  const sendMessage = async () => {
    const content = input.trim();
    if (!content) return;
    setError(null);
    setLatency(null);
    setInput("");

    const history = messages;
    setMessages((prev) => [...prev, { role: "user", content }]);
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: content, history }),
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "Request failed");
      }

      const data = await response.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.response || "" },
      ]);
      if (typeof data.latency_ms === "number") {
        setLatency(data.latency_ms);
      }
    } catch (err) {
      setError(err.message);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "서버에 연결할 수 없습니다. API 주소를 확인해 주세요.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (canSend) sendMessage();
    }
  };

  const resetChat = () => {
    setMessages(starter);
    setLatency(null);
    setError(null);
  };

  const status = useMemo(() => {
    if (loading) return "응답 생성 중";
    if (error) return "연결 끊김";
    return "온라인";
  }, [loading, error]);

  return (
    <div className="app">
      <div className="backdrop" />
      <header className="topbar">
        <div>
          <p className="eyebrow">Likelion 실전 프로젝트 2</p>
          <h1>QA 챗봇 운영 대시보드</h1>
          <p className="subhead">
            Kanana 8B RLHF 기반 응답 품질을 실시간으로 확인합니다.
          </p>
        </div>
        <div className="status-card">
          <p className="status-label">세션</p>
          <p className="status-value">{status}</p>
          <p className="status-meta">메시지: {messageCount}</p>
          {latency !== null && (
            <p className="status-meta">지연: {latency.toFixed(0)} ms</p>
          )}
        </div>
      </header>

      <main className="chat-shell">
        <div className="chat-header">
          <div>
            <h2>대화</h2>
            <p>API base: {API_BASE}</p>
          </div>
          <button className="ghost" onClick={resetChat} type="button">
            초기화
          </button>
        </div>

        <section className="messages">
          {messages.map((message, index) => (
            <div
              key={`${message.role}-${index}`}
              className={`bubble ${message.role}`}
            >
              <span className="role">{message.role}</span>
              <p>{message.content}</p>
            </div>
          ))}
          {loading && (
            <div className="bubble assistant">
              <span className="role">assistant</span>
              <p className="typing">
                생각 중<span>.</span>
                <span>.</span>
                <span>.</span>
              </p>
            </div>
          )}
        </section>

        <section className="composer">
          <textarea
            placeholder="메시지를 입력하세요. Enter로 전송, Shift+Enter로 줄바꿈."
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleKeyDown}
          />
          <div className="composer-actions">
            {error && <span className="error">{error}</span>}
            <button className="primary" onClick={sendMessage} disabled={!canSend}>
              {loading ? "전송 중..." : "전송"}
            </button>
          </div>
        </section>
      </main>
    </div>
  );
}
