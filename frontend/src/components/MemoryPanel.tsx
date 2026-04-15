import type { MemoryItem } from "../types";

interface Props {
  memories: MemoryItem[];
  onClearMemory: () => void;
}

export default function MemoryPanel({ memories, onClearMemory }: Props) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between">
        <div className="text-base font-semibold text-slate-900">Long-term Memory</div>
        <button
          onClick={onClearMemory}
          className="rounded-xl border border-slate-300 bg-white px-3 py-1 text-sm hover:bg-slate-50"
        >
          清空记忆
        </button>
      </div>

      {memories.length === 0 ? (
        <div className="text-sm text-slate-500">暂无长期记忆</div>
      ) : (
        <div className="space-y-3">
          {memories.map((item) => (
            <div
              key={item.memory_id}
              className="rounded-xl border border-slate-200 bg-slate-50 p-3"
            >
              <div className="mb-2 flex flex-wrap gap-2 text-xs text-slate-600">
                <span className="rounded bg-white px-2 py-1">{item.memory_type}</span>
                <span className="rounded bg-white px-2 py-1">{item.importance}</span>
                {item.source_session_id && (
                  <span className="rounded bg-white px-2 py-1">
                    {item.source_session_id}
                  </span>
                )}
              </div>

              <div className="mb-2 text-sm leading-6 text-slate-800">{item.content}</div>

              {item.source_query && (
                <div className="text-xs text-slate-500">source_query: {item.source_query}</div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}