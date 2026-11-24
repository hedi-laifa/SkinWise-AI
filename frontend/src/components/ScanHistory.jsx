/**
 * ScanHistory — thumbnails of the last analyses in this session; click to restore one.
 */
import { DEFAULT_META, SEVERITY_META } from "../constants/severity.js";

export default function ScanHistory({ items, onSelect }) {
  if (!items || items.length === 0) return null;

  return (
    <div className="pt-3 border-t border-slate-100 space-y-2">
      <p className="text-[11px] font-semibold uppercase tracking-widest text-slate-400">Session History ({items.length})</p>
      <div className="flex gap-2 overflow-x-auto pb-1">
        {items.map((item) => {
          const m = SEVERITY_META[item.prediction.predicted_class] || DEFAULT_META;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => onSelect(item)}
              className="shrink-0 flex items-center gap-2 p-1.5 rounded-lg bg-slate-50 border border-slate-200 hover:border-blue-200 hover:bg-blue-50 transition text-left cursor-pointer"
            >
              {item.previewUrl && <img src={item.previewUrl} alt={item.fileName} className="w-9 h-9 rounded-md object-cover border border-slate-200" />}
              <div>
                <div className="flex items-center gap-1">
                  <span className={`h-1.5 w-1.5 rounded-full ${m.dot}`} />
                  <span className="text-[11px] font-semibold text-slate-700">{m.shortLabel}</span>
                </div>
                <span className="text-[10px] font-mono text-slate-400">{item.timestamp}</span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
