/**
 * SeverityCard — displays one ACNE04 severity class with its metadata.
 * Highlights when it matches the current prediction.
 */
export default function SeverityCard({ clsName, meta, isPredicted }) {
  return (
    <div
      className={`rounded-xl border p-3.5 transition-all ${
        isPredicted
          ? `${meta.cardBorder} ring-1 ring-white/20 shadow-lg`
          : "border-slate-800/80 bg-slate-900/50 hover:bg-slate-900/80"
      }`}
    >
      <div className="flex items-center justify-between mb-1">
        <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-slate-200">
          <span className={`h-2 w-2 rounded-full ${meta.dot}`} />
          {meta.shortLabel}
        </span>
        <span className="text-[11px] font-mono text-slate-400 px-1.5 py-0.5 rounded bg-slate-800/80">
          {meta.level}
        </span>
      </div>
      <p className="text-[11px] font-mono text-slate-400">{meta.lesionRange}</p>
    </div>
  );
}
