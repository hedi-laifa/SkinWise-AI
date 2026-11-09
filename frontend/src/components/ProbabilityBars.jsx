/**
 * ProbabilityBars — animated horizontal probability bars for each ACNE04 class.
 */
export default function ProbabilityBars({ entries, predictedClass, severityMeta }) {
  if (!entries || entries.length === 0) return null;

  return (
    <div className="space-y-3">
      {entries.map(([cls, prob]) => {
        const meta = severityMeta[cls];
        const pct = (prob * 100).toFixed(1);
        const isTop = cls === predictedClass;
        return (
          <div key={cls}>
            <div className="flex items-center justify-between mb-1">
              <span
                className={`text-xs font-medium ${
                  isTop ? "text-white" : "text-slate-400"
                }`}
              >
                {meta?.shortLabel ?? cls}
              </span>
              <span
                className={`text-xs font-mono tabular-nums ${
                  isTop ? "text-white font-semibold" : "text-slate-400"
                }`}
              >
                {pct}%
              </span>
            </div>
            <div className="h-2 w-full rounded-full bg-slate-800/80 overflow-hidden">
              <div
                className={`h-full rounded-full bg-gradient-to-r transition-all duration-700 ease-out ${
                  meta?.barGradient ?? "from-slate-500 to-slate-400"
                }`}
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
