/**
 * ProbabilityBars — animated softmax probability bar chart for the 4 ACNE04 classes.
 */
import { CLASS_ORDER, DEFAULT_META, SEVERITY_META } from "../constants/severity.js";

export default function ProbabilityBars({ probabilities, predictedClass }) {
  if (!probabilities) return null;
  const entries = CLASS_ORDER.filter((c) => c in probabilities).map((c) => [c, probabilities[c]]);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-[11px] font-semibold uppercase tracking-widest text-slate-400">
        <span>Softmax Probabilities</span>
        <span>%</span>
      </div>
      <div className="space-y-2">
        {entries.map(([cls, prob]) => {
          const pct = Math.max(0, Math.min(100, Number(prob) * 100));
          const winner = cls === predictedClass;
          const m = SEVERITY_META[cls] || DEFAULT_META;
          return (
            <div key={cls} className={`p-3 rounded-lg border transition-all ${
              winner ? "border-blue-200 bg-blue-50" : "border-slate-100 bg-slate-50"
            }`}>
              <div className="flex items-center justify-between text-xs mb-1.5">
                <div className="flex items-center gap-2">
                  <span className={`h-2 w-2 rounded-full ${m.dot}`} />
                  <span className={winner ? "font-bold text-slate-900" : "font-medium text-slate-600"}>{cls}</span>
                  <span className="hidden sm:inline text-[10px] font-mono text-slate-400">({m.lesionRange})</span>
                  {winner && (
                    <span className="text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded bg-blue-100 text-blue-700 border border-blue-200">
                      Predicted
                    </span>
                  )}
                </div>
                <span className="font-mono font-bold text-slate-700">{pct.toFixed(2)}%</span>
              </div>
              <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full ${m.barColor} transition-all duration-500 ease-out`}
                  style={{ width: `${pct}%` }}
                  role="progressbar"
                  aria-valuenow={Number(pct.toFixed(2))}
                  aria-valuemin={0}
                  aria-valuemax={100}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
