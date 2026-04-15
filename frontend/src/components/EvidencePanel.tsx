import { useState } from "react";
import type { RetrievalResult } from "../types";

interface Props {
  evidences: RetrievalResult[];
}

export default function EvidencePanel({ evidences }: Props) {
  const [expandedIds, setExpandedIds] = useState<Record<string, boolean>>({});

  function toggle(key: string) {
    setExpandedIds((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  }

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-3 text-base font-semibold text-slate-900">Evidence</div>

      {evidences.length === 0 ? (
        <div className="text-sm text-slate-500">暂无检索证据</div>
      ) : (
        <div className="space-y-3">
          {evidences.map((item, index) => {
            const key = `${item.chunk_id}-${index}`;
            const expanded = !!expandedIds[key];
            const preview = expanded ? item.preview : item.preview.slice(0, 150);

            return (
              <div
                key={key}
                className="rounded-xl border border-slate-200 bg-slate-50 p-3"
              >
                <div className="mb-2 flex flex-wrap gap-2 text-xs text-slate-600">
                  <span className="rounded bg-white px-2 py-1">{item.source}</span>
                  <span className="rounded bg-white px-2 py-1">
                    chunk: {item.chunk_index}
                  </span>
                  <span className="rounded bg-white px-2 py-1">
                    distance: {item.distance}
                  </span>
                </div>

                <div className="text-sm leading-6 text-slate-800">
                  {preview}
                  {!expanded && item.preview.length > 150 ? "..." : ""}
                </div>

                {item.preview.length > 150 && (
                  <button
                    onClick={() => toggle(key)}
                    className="mt-2 text-xs font-medium text-slate-600 hover:text-slate-900"
                  >
                    {expanded ? "收起" : "展开"}
                  </button>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}