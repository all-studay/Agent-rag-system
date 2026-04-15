import { useState } from "react";
import type { ChatMessage } from "../types";

interface Props {
  message: ChatMessage;
}

function confidenceColor(confidence?: string) {
  switch (confidence) {
    case "high":
      return "bg-emerald-100 text-emerald-700";
    case "medium":
      return "bg-amber-100 text-amber-700";
    case "low":
      return "bg-rose-100 text-rose-700";
    default:
      return "bg-slate-100 text-slate-700";
  }
}

function formatTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleTimeString();
}

export default function ChatMessageCard({ message }: Props) {
  const isUser = message.role === "user";
  const [copied, setCopied] = useState(false);

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div
      className={`rounded-2xl border p-4 shadow-sm ${
        isUser
          ? "border-slate-200 bg-slate-900 text-white"
          : "border-slate-200 bg-white text-slate-900"
      }`}
    >
      <div className="mb-3 flex items-center justify-between">
        <div className="text-sm font-semibold">{isUser ? "用户" : "助手"}</div>
        <div className={`text-xs ${isUser ? "text-slate-300" : "text-slate-500"}`}>
          {formatTime(message.createdAt)}
        </div>
      </div>

      <div className="whitespace-pre-wrap text-sm leading-7">{message.content}</div>

      {!isUser && (
        <div className="mt-3 flex items-center gap-2">
          <button
            onClick={handleCopy}
            className="rounded-lg border border-slate-300 bg-white px-3 py-1 text-xs text-slate-700 hover:bg-slate-50"
          >
            {copied ? "已复制" : "复制回答"}
          </button>
        </div>
      )}

      {!isUser && message.meta && (
        <div className="mt-4 space-y-3 border-t border-slate-200 pt-4">
          <div className="flex flex-wrap gap-2">
            <span
              className={`rounded-full px-3 py-1 text-xs font-medium ${confidenceColor(
                message.meta.confidence
              )}`}
            >
              confidence: {message.meta.confidence}
            </span>
            <span
              className={`rounded-full px-3 py-1 text-xs font-medium ${
                message.meta.is_answerable
                  ? "bg-blue-100 text-blue-700"
                  : "bg-slate-200 text-slate-700"
              }`}
            >
              is_answerable: {String(message.meta.is_answerable)}
            </span>
          </div>

          <div>
            <div className="mb-1 text-xs font-semibold text-slate-500">Sources</div>
            <div className="flex flex-wrap gap-2">
              {message.meta.sources.map((source) => (
                <span
                  key={source}
                  className="rounded-full bg-slate-100 px-3 py-1 text-xs text-slate-700"
                >
                  {source}
                </span>
              ))}
            </div>
          </div>

          {message.meta.refusal_reason && (
            <div>
              <div className="mb-1 text-xs font-semibold text-slate-500">
                Refusal Reason
              </div>
              <div className="text-sm text-rose-600">{message.meta.refusal_reason}</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}