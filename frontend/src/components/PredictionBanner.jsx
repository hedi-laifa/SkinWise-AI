/**
 * PredictionBanner — predicted grade, clinical description and a confidence ring.
 */
import { DEFAULT_META, SEVERITY_META } from "../constants/severity.js";

const RING_R = 34;
const RING_C = 2 * Math.PI * RING_R;

export default function PredictionBanner({ prediction }) {
  const meta = SEVERITY_META[prediction.predicted_class] || DEFAULT_META;
  const topPct = (
    (prediction.confidence ?? prediction.probabilities?.[prediction.predicted_class] ?? 0) * 100
  ).toFixed(1);
  const ringOffset = RING_C - (Number(topPct) / 100) * RING_C;

  return (
    <div className={`rounded-xl border p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4 ${meta.cardCls}`}>
      <div className="space-y-2">
        <span className={`inline-flex items-center gap-1.5 text-[11px] font-semibold uppercase px-2.5 py-0.5 rounded-full border ${meta.badgeCls}`}>
          <span className={`h-1.5 w-1.5 rounded-full ${meta.dot}`} />
          {meta.level} · {meta.lesionRange}
        </span>
        <h3 className="text-2xl font-bold text-slate-900">{prediction.predicted_class}</h3>
        <p className="text-xs text-slate-600 leading-relaxed max-w-sm">{meta.description}</p>
        <div className="flex items-start gap-2 text-xs text-slate-600">
          <span className="shrink-0 font-semibold text-[10px] uppercase px-1.5 py-0.5 rounded bg-white/70 border border-slate-200 text-slate-500">
            Clinical Note
          </span>
          <span>{meta.recommendation}</span>
        </div>
      </div>

      {/* Confidence ring */}
      <div className="shrink-0 flex flex-col items-center gap-1 bg-white/70 border border-white rounded-xl px-5 py-3">
        <div className="relative w-16 h-16">
          <svg className="w-16 h-16 -rotate-90" viewBox="0 0 80 80">
            <circle cx="40" cy="40" r={RING_R} stroke="#e2e8f0" strokeWidth="7" fill="transparent" />
            <circle cx="40" cy="40" r={RING_R} stroke={meta.ringColor} strokeWidth="7"
              strokeDasharray={RING_C} strokeDashoffset={ringOffset} strokeLinecap="round" fill="transparent"
              className="transition-all duration-700 ease-out" />
          </svg>
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-sm font-bold font-mono text-slate-800">{topPct}%</span>
          </div>
        </div>
        <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wide">Confidence</span>
      </div>
    </div>
  );
}
