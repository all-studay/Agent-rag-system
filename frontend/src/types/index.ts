export interface RetrievalResult {
  source: string;
  chunk_id: string;
  chunk_index: number;
  distance: number;
  preview: string;
}

export interface QAResponse {
  query: string;
  answer: string;
  is_answerable: boolean;
  confidence: "high" | "medium" | "low";
  sources: string[];
  retrieval_results: RetrievalResult[];
  refusal_reason: string | null;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: string;
  meta?: QAResponse;
}

export interface SessionDebugInfo {
  session_id: string;
  history_count: number;
  raw_history: Array<{
    user_query: string;
    assistant_answer: string;
  }>;
  raw_history_text: string;
  conversation_summary: string;
  recent_history: string;
}

export interface MemoryItem {
  memory_id: string;
  memory_type: string;
  content: string;
  importance: string;
  source_session_id?: string | null;
  source_query?: string | null;
  created_at: string;
}

export interface MemoryListResponse {
  count: number;
  memories: MemoryItem[];
}

export interface SessionRecord {
  sessionId: string;
  title: string;
  updatedAt: string;
  messages: ChatMessage[];
}