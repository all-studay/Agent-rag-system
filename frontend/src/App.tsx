import { useEffect, useMemo, useRef, useState } from "react";
import {
  askQuestion,
  clearMemory,
  clearSession,
  getMemoryList,
  getSessionDebug,
} from "./api/client";
import ChatInput from "./components/ChatInput";
import ChatMessageCard from "./components/ChatMessageCard";
import EvidencePanel from "./components/EvidencePanel";
import LoadingBadge from "./components/LoadingBadge";
import MemoryPanel from "./components/MemoryPanel";
import SessionDebugPanel from "./components/SessionDebugPanel";
import SessionSidebar from "./components/SessionSidebar";
import SessionControlBar from "./components/SessionControlBar";
import type {
  ChatMessage,
  MemoryItem,
  QAResponse,
  RetrievalResult,
  SessionDebugInfo,
  SessionRecord,
} from "./types";

type RightTab = "evidence" | "debug" | "memory";

const STORAGE_KEY = "agent-rag-demo-sessions";
const ACTIVE_SESSION_KEY = "agent-rag-demo-active-session";

function createSessionId() {
  return `session-${Date.now()}`;
}

function createId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function nowIso() {
  return new Date().toISOString();
}

function safeLoadSessions(): SessionRecord[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    return JSON.parse(raw) as SessionRecord[];
  } catch {
    return [];
  }
}

function safeLoadActiveSessionId(): string | null {
  try {
    return localStorage.getItem(ACTIVE_SESSION_KEY);
  } catch {
    return null;
  }
}

export default function App() {
  const [sessions, setSessions] = useState<SessionRecord[]>([]);
  const [sessionId, setSessionId] = useState("");
  const [loading, setLoading] = useState(false);
  const [debugInfo, setDebugInfo] = useState<SessionDebugInfo | null>(null);
  const [memories, setMemories] = useState<MemoryItem[]>([]);
  const [error, setError] = useState("");
  const [rightTab, setRightTab] = useState<RightTab>("evidence");

  const messageScrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const loadedSessions = safeLoadSessions();
    const activeId = safeLoadActiveSessionId();

    if (loadedSessions.length > 0) {
      setSessions(loadedSessions);

      const matched =
        loadedSessions.find((s) => s.sessionId === activeId) ?? loadedSessions[0];
      setSessionId(matched.sessionId);
    } else {
      const newSession: SessionRecord = {
        sessionId: createSessionId(),
        title: "新会话",
        updatedAt: nowIso(),
        messages: [],
      };
      setSessions([newSession]);
      setSessionId(newSession.sessionId);
    }
  }, []);

  useEffect(() => {
    if (sessions.length > 0) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
    }
  }, [sessions]);

  useEffect(() => {
    if (sessionId) {
      localStorage.setItem(ACTIVE_SESSION_KEY, sessionId);
    }
  }, [sessionId]);

  useEffect(() => {
    if (messageScrollRef.current) {
      messageScrollRef.current.scrollTop = messageScrollRef.current.scrollHeight;
    }
  }, [sessionId, sessions, loading]);

  const currentSession = useMemo(
    () => sessions.find((s) => s.sessionId === sessionId) ?? null,
    [sessions, sessionId]
  );

  const messages = currentSession?.messages ?? [];

  const latestAssistantMeta = useMemo<QAResponse | undefined>(() => {
    const assistantMessages = messages.filter((m) => m.role === "assistant");
    return assistantMessages.length
      ? assistantMessages[assistantMessages.length - 1].meta
      : undefined;
  }, [messages]);

  const evidences: RetrievalResult[] = latestAssistantMeta?.retrieval_results ?? [];

  function updateCurrentSession(updater: (session: SessionRecord) => SessionRecord) {
    setSessions((prev) =>
      prev.map((session) =>
        session.sessionId === sessionId ? updater(session) : session
      )
    );
  }

  async function handleSend(text: string) {
    if (!currentSession) return;

    setError("");
    const createdAt = nowIso();

    const userMessage: ChatMessage = {
      id: createId(),
      role: "user",
      content: text,
      createdAt,
    };

    updateCurrentSession((session) => ({
      ...session,
      title: session.messages.length === 0 ? text.slice(0, 20) || "新会话" : session.title,
      updatedAt: createdAt,
      messages: [...session.messages, userMessage],
    }));

    setLoading(true);

    try {
      const result = await askQuestion({
        query: text,
        top_k: 3,
        session_id: sessionId,
      });

      const assistantMessage: ChatMessage = {
        id: createId(),
        role: "assistant",
        content: result.answer,
        createdAt: nowIso(),
        meta: result,
      };

      updateCurrentSession((session) => ({
        ...session,
        updatedAt: nowIso(),
        messages: [...session.messages, assistantMessage],
      }));

      const [debug, memoryResp] = await Promise.all([
        getSessionDebug(sessionId),
        getMemoryList(),
      ]);

      setDebugInfo(debug);
      setMemories(memoryResp.memories);
      setRightTab("evidence");
    } catch (err: any) {
      setError(err?.response?.data?.detail || err?.message || "请求失败");
    } finally {
      setLoading(false);
    }
  }

  async function handleNewSession() {
    const newSession: SessionRecord = {
      sessionId: createSessionId(),
      title: "新会话",
      updatedAt: nowIso(),
      messages: [],
    };
    setSessions((prev) => [newSession, ...prev]);
    setSessionId(newSession.sessionId);
    setDebugInfo(null);
    setError("");
  }

  async function handleDeleteCurrentSession() {
    if (!currentSession) return;

    try {
      await clearSession(currentSession.sessionId);
    } catch {
      // 后端失败时仍允许本地删除
    }

    setSessions((prev) => {
      const filtered = prev.filter((s) => s.sessionId !== currentSession.sessionId);

      if (filtered.length === 0) {
        const fallback: SessionRecord = {
          sessionId: createSessionId(),
          title: "新会话",
          updatedAt: nowIso(),
          messages: [],
        };
        setSessionId(fallback.sessionId);
        return [fallback];
      }

      setSessionId(filtered[0].sessionId);
      return filtered;
    });

    setDebugInfo(null);
    setError("");
  }

  async function handleClearSession() {
    try {
      await clearSession(sessionId);
      updateCurrentSession((session) => ({
        ...session,
        title: "新会话",
        updatedAt: nowIso(),
        messages: [],
      }));
      setDebugInfo(null);
      setError("");
    } catch (err: any) {
      setError(err?.response?.data?.detail || err?.message || "清空会话失败");
    }
  }

  async function handleRefreshDebug() {
    try {
      const data = await getSessionDebug(sessionId);
      setDebugInfo(data);
      setRightTab("debug");
      setError("");
    } catch (err: any) {
      setError(err?.response?.data?.detail || err?.message || "获取 session debug 失败");
    }
  }

  async function handleRefreshMemory() {
    try {
      const data = await getMemoryList();
      setMemories(data.memories);
      setRightTab("memory");
      setError("");
    } catch (err: any) {
      setError(err?.response?.data?.detail || err?.message || "获取 memory 失败");
    }
  }

  async function handleClearMemory() {
    try {
      await clearMemory();
      await handleRefreshMemory();
    } catch (err: any) {
      setError(err?.response?.data?.detail || err?.message || "清空 memory 失败");
    }
  }

  return (
    <div className="h-screen overflow-hidden bg-slate-100">
      <div className="mx-auto flex h-full max-w-[1700px] flex-col p-6">
        <div className="mb-4 shrink-0">
          <h1 className="text-3xl font-bold text-slate-900">Agent RAG System Demo</h1>
          <p className="mt-2 text-sm text-slate-600">
            展示混合检索、结构化问答、多轮上下文、历史摘要与长期记忆能力
          </p>
        </div>

        {error && (
          <div className="mb-4 shrink-0 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
            {error}
          </div>
        )}

        <div className="min-h-0 flex-1">
          <div className="grid h-full grid-cols-1 gap-6 xl:grid-cols-[280px_1.2fr_0.9fr]">
            {/* 左栏：会话列表独立滚动 */}
            <div className="min-h-0">
              <SessionSidebar
                sessions={sessions}
                activeSessionId={sessionId}
                onSelect={setSessionId}
                onCreate={handleNewSession}
                onDeleteCurrent={handleDeleteCurrentSession}
              />
            </div>

            {/* 中栏：聊天区独立滚动 */}
            <div className="flex min-h-0 flex-col gap-4">
              <div className="shrink-0">
                <SessionControlBar
                  sessionId={sessionId}
                  onSessionIdChange={setSessionId}
                  onNewSession={handleNewSession}
                  onClearSession={handleClearSession}
                  onRefreshDebug={handleRefreshDebug}
                  onRefreshMemory={handleRefreshMemory}
                />
              </div>

              <div className="flex min-h-0 flex-1 flex-col rounded-2xl border border-slate-200 bg-slate-50 p-4">
                <div className="mb-3 flex shrink-0 items-center justify-between">
                  <div className="text-sm font-medium text-slate-600">
                    当前会话：{sessionId}
                  </div>
                  {loading && <LoadingBadge />}
                </div>

                <div
                  ref={messageScrollRef}
                  className="min-h-0 flex-1 space-y-4 overflow-auto pr-1"
                >
                  {messages.length === 0 ? (
                    <div className="rounded-2xl border border-dashed border-slate-300 bg-white p-8 text-center text-slate-500">
                      还没有消息，先发一个问题试试。
                    </div>
                  ) : (
                    messages.map((msg) => <ChatMessageCard key={msg.id} message={msg} />)
                  )}
                </div>
              </div>

              <div className="shrink-0">
                <ChatInput loading={loading} onSend={handleSend} />
              </div>
            </div>

            {/* 右栏：调试区独立滚动 */}
            <div className="flex min-h-0 flex-col gap-4">
              <div className="shrink-0 rounded-2xl border border-slate-200 bg-white p-2 shadow-sm">
                <div className="grid grid-cols-3 gap-2">
                  <button
                    onClick={() => setRightTab("evidence")}
                    className={`rounded-xl px-3 py-2 text-sm ${
                      rightTab === "evidence"
                        ? "bg-slate-900 text-white"
                        : "bg-slate-100 text-slate-700"
                    }`}
                  >
                    Evidence
                  </button>
                  <button
                    onClick={() => setRightTab("debug")}
                    className={`rounded-xl px-3 py-2 text-sm ${
                      rightTab === "debug"
                        ? "bg-slate-900 text-white"
                        : "bg-slate-100 text-slate-700"
                    }`}
                  >
                    Session
                  </button>
                  <button
                    onClick={() => setRightTab("memory")}
                    className={`rounded-xl px-3 py-2 text-sm ${
                      rightTab === "memory"
                        ? "bg-slate-900 text-white"
                        : "bg-slate-100 text-slate-700"
                    }`}
                  >
                    Memory
                  </button>
                </div>
              </div>

              <div className="min-h-0 flex-1 overflow-auto pr-1">
                {rightTab === "evidence" && <EvidencePanel evidences={evidences} />}
                {rightTab === "debug" && <SessionDebugPanel debugInfo={debugInfo} />}
                {rightTab === "memory" && (
                  <MemoryPanel memories={memories} onClearMemory={handleClearMemory} />
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}