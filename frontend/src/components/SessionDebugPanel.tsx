import type { SessionDebugInfo } from "../types";

interface Props {
  debugInfo: SessionDebugInfo | null;
}

export default function SessionDebugPanel({ debugInfo }: Props) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-3 text-base font-semibold text-slate-900">Session Debug</div>

      {!debugInfo ? (
        <div className="text-sm text-slate-500">暂无 session debug 信息</div>
      ) : (
        <div className="space-y-4 text-sm">
          <div>
            <div className="mb-1 font-medium text-slate-700">Session ID</div>
            <div className="rounded-xl bg-slate-50 p-3">{debugInfo.session_id}</div>
          </div>

          <div>
            <div className="mb-1 font-medium text-slate-700">History Count</div>
            <div className="rounded-xl bg-slate-50 p-3">{debugInfo.history_count}</div>
          </div>

          <div>
            <div className="mb-1 font-medium text-slate-700">Conversation Summary</div>
            <div className="whitespace-pre-wrap rounded-xl bg-slate-50 p-3 leading-6">
              {debugInfo.conversation_summary}
            </div>
          </div>

          <div>
            <div className="mb-1 font-medium text-slate-700">Recent History</div>
            <div className="whitespace-pre-wrap rounded-xl bg-slate-50 p-3 leading-6">
              {debugInfo.recent_history}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}