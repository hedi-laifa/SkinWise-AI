/**
 * SeverityCard — one ACNE04 severity grade in the reference strip.
 * Highlighted when it matches the current prediction.
 */
export default function SeverityCard({ meta, isPredicted }) {
  return (
    <div className={`rounded-lg border p-3 transition-all ${
      isPredicted ? `${meta.cardCls} shadow-sm` : "bg-white border-slate-200"
    }`}>
      <div className="flex items-center justify-between mb-1">
        <span className="flex items-center gap-1.5 text-xs font-semibold text-slate-700">
          <span className={`h-2 w-2 rounded-full ${meta.dot}`} />
          {meta.shortLabel}
        </span>
        <span className="text-[11px] font-mono text-slate-400 px-1.5 py-0.5 rounded bg-slate-100">
          {meta.level}
        </span>
      </div>
      <p className="text-[11px] text-slate-500 font-mono">{meta.lesionRange}</p>
    </div>
  );
}
