import { useState } from "react";

interface Props {
  loading: boolean;
  onSend: (text: string) => Promise<void>;
}

export default function ChatInput({ loading, onSend }: Props) {
  const [value, setValue] = useState("");

  async function handleSend() {
    const text = value.trim();
    if (!text || loading) return;

    setValue("");
    await onSend(text);
  }

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-2 text-sm font-medium text-slate-700">输入问题</div>
      <div className="flex gap-3">
        <textarea
          rows={4}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
              void handleSend();
            }
          }}
          placeholder="请输入你的问题，Ctrl/Cmd + Enter 发送"
          className="flex-1 resize-none rounded-xl border border-slate-300 px-3 py-3 outline-none focus:border-slate-500"
        />
        <button
          onClick={handleSend}
          disabled={loading}
          className="min-w-[110px] rounded-xl bg-slate-900 px-4 py-2 text-white hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-400"
        >
          {loading ? "发送中..." : "发送"}
        </button>
      </div>
    </div>
  );
}