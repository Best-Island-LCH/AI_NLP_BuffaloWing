import { useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "https://nontheatrical-judiciarily-susanne.ngrok-free.dev";

const starter = [
  {
    role: "assistant",
    content:
      "Ready when you are. Ask anything, and I will route it through the pipeline.",
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
          content: "Sorry, I could not reach the server. Check the API base.",
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
    if (loading) return "Streaming...";
    if (error) return "Disconnected";
    return "Online";
  }, [loading, error]);

  return (
    <div className="app">
      <div className="backdrop" />
      <header className="topbar">
        <div>
          <p className="eyebrow">LLM Orchestrator</p>
          <h1>Chatbot Control Room</h1>
          <p className="subhead">
            Connects your FastAPI gateway to the pipeline in real time.
          </p>
        </div>
        <div className="status-card">
          <p className="status-label">Session</p>
          <p className="status-value">{status}</p>
          <p className="status-meta">Messages: {messageCount}</p>
          {latency !== null && (
            <p className="status-meta">Latency: {latency.toFixed(0)} ms</p>
          )}
        </div>
      </header>

      <main className="chat-shell">
        <div className="chat-header">
          <div>
            <h2>Conversation</h2>
            <p>API base: {API_BASE}</p>
          </div>
          <button className="ghost" onClick={resetChat} type="button">
            Reset
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
                Thinking<span>.</span>
                <span>.</span>
                <span>.</span>
              </p>
            </div>
          )}
        </section>

        <section className="composer">
          <textarea
            placeholder="Write a message. Press Enter to send, Shift+Enter for a new line."
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleKeyDown}
          />
          <div className="composer-actions">
            {error && <span className="error">{error}</span>}
            <button className="primary" onClick={sendMessage} disabled={!canSend}>
              {loading ? "Sending..." : "Send"}
            </button>
          </div>
        </section>
      </main>
    </div>
  );
}
