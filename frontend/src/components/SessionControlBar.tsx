interface Props {
  sessionId: string;
  onSessionIdChange: (value: string) => void;
  onNewSession: () => void;
  onClearSession: () => void;
  onRefreshDebug: () => void;
  onRefreshMemory: () => void;
}

export default function SessionControlBar({
  sessionId,
  onSessionIdChange,
  onNewSession,
  onClearSession,
  onRefreshDebug,
  onRefreshMemory,
}: Props) {
  return (
    <div className="flex flex-wrap items-center gap-3 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex min-w-[260px] flex-col gap-1">
        <label className="text-sm font-medium text-slate-700">Session ID</label>
        <input
          value={sessionId}
          onChange={(e) => onSessionIdChange(e.target.value)}
          className="rounded-xl border border-slate-300 px-3 py-2 outline-none focus:border-slate-500"
          placeholder="请输入 session id"
        />
      </div>

      <div className="flex flex-wrap gap-2">
        <button
          onClick={onNewSession}
          className="rounded-xl bg-slate-900 px-4 py-2 text-white hover:bg-slate-700"
        >
          新建会话
        </button>
        <button
          onClick={onClearSession}
          className="rounded-xl border border-slate-300 bg-white px-4 py-2 hover:bg-slate-50"
        >
          清空会话
        </button>
        <button
          onClick={onRefreshDebug}
          className="rounded-xl border border-slate-300 bg-white px-4 py-2 hover:bg-slate-50"
        >
          刷新 Session Debug
        </button>
        <button
          onClick={onRefreshMemory}
          className="rounded-xl border border-slate-300 bg-white px-4 py-2 hover:bg-slate-50"
        >
          刷新 Memory
        </button>
      </div>
    </div>
  );
}