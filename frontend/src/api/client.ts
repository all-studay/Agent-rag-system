import axios from "axios";
import type { MemoryListResponse, QAResponse, SessionDebugInfo } from "../types";

const api = axios.create({
  baseURL: "http://127.0.0.1:8000",
  timeout: 180000,
});

export async function askQuestion(params: {
  query: string;
  top_k?: number;
  session_id?: string;
}): Promise<QAResponse> {
  const { data } = await api.post<QAResponse>("/qa", {
    query: params.query,
    top_k: params.top_k ?? 3,
    session_id: params.session_id,
  });
  return data;
}

export async function clearSession(sessionId: string): Promise<{ message: string }> {
  const { data } = await api.post<{ message: string }>("/session/clear", {
    session_id: sessionId,
  });
  return data;
}

export async function getSessionDebug(sessionId: string): Promise<SessionDebugInfo> {
  const { data } = await api.get<SessionDebugInfo>("/session/debug", {
    params: { session_id: sessionId },
  });
  return data;
}

export async function getMemoryList(): Promise<MemoryListResponse> {
  const { data } = await api.get<MemoryListResponse>("/memory/list");
  return data;
}

export async function clearMemory(): Promise<{ message: string }> {
  const { data } = await api.post<{ message: string }>("/memory/clear");
  return data;
}