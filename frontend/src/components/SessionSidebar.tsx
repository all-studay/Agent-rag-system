import type { SessionRecord } from "../types";

interface Props {
  sessions: SessionRecord[];
  activeSessionId: string;
  onSelect: (sessionId: string) => void;
  onCreate: () => void;
  onDeleteCurrent: () => void;
}

function formatTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

export default function SessionSidebar({
  sessions,
  activeSessionId,
  onSelect,
  onCreate,
  onDeleteCurrent,
}: Props) {
  return (
    <div className="flex h-full min-h-0 flex-col rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <div className="text-base font-semibold text-slate-900">会话列表</div>
          <div className="text-xs text-slate-500">本地持久化保存</div>
        </div>
        <button
          onClick={onCreate}
          className="rounded-xl bg-slate-900 px-3 py-2 text-sm text-white hover:bg-slate-700"
        >
          新建
        </button>
      </div>

      <div className="mb-3">
        <button
          onClick={onDeleteCurrent}
          className="w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm hover:bg-slate-50"
        >
          删除当前会话
        </button>
      </div>

      <div className="min-h-0 flex-1 space-y-3 overflow-auto">
        {sessions.length === 0 ? (
          <div className="rounded-xl border border-dashed border-slate-300 p-4 text-sm text-slate-500">
            暂无会话，点击“新建”开始。
          </div>
        ) : (
          sessions.map((session) => {
            const active = session.sessionId === activeSessionId;
            return (
              <button
                key={session.sessionId}
                onClick={() => onSelect(session.sessionId)}
                className={`w-full rounded-2xl border p-3 text-left transition ${
                  active
                    ? "border-slate-900 bg-slate-900 text-white"
                    : "border-slate-200 bg-slate-50 hover:bg-slate-100"
                }`}
              >
                <div className="truncate text-sm font-semibold">
                  {session.title || session.sessionId}
                </div>
                <div
                  className={`mt-1 text-xs ${
                    active ? "text-slate-200" : "text-slate-500"
                  }`}
                >
                  {session.sessionId}
                </div>
                <div
                  className={`mt-2 text-xs ${
                    active ? "text-slate-300" : "text-slate-400"
                  }`}
                >
                  {formatTime(session.updatedAt)}
                </div>
              </button>
            );
          })
        )}
      </div>
    </div>
  );
}