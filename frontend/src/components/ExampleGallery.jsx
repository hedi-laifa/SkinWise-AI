/**
 * ExampleGallery — preloaded test images (one per grade): one click runs a full analysis.
 * The images come from the held-out TEST split, so the model never trained on them.
 */
import { SEVERITY_META } from "../constants/severity.js";

const EXAMPLES = [
  { src: "/examples/mild.jpg", fileName: "example_mild.jpg", trueClass: "Mild (Level 0)" },
  { src: "/examples/moderate.jpg", fileName: "example_moderate.jpg", trueClass: "Moderate (Level 1)" },
  { src: "/examples/severe.jpg", fileName: "example_severe.jpg", trueClass: "Severe (Level 2)" },
  { src: "/examples/very_severe.jpg", fileName: "example_very_severe.jpg", trueClass: "Very Severe (Level 3)" },
];

export default function ExampleGallery({ onSelect, isLoading }) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-[11px] font-semibold uppercase tracking-widest text-slate-400">Or try an example</p>
        <p className="text-[10px] text-slate-400">Test-set images · true grade shown</p>
      </div>
      <div className="grid grid-cols-4 gap-2">
        {EXAMPLES.map((example) => {
          const meta = SEVERITY_META[example.trueClass];
          return (
            <button
              key={example.src}
              type="button"
              disabled={isLoading}
              onClick={() => onSelect(example)}
              title={`Analyze an example labelled ${example.trueClass}`}
              className="group rounded-lg border border-slate-200 bg-white overflow-hidden text-left transition hover:border-blue-300 hover:shadow-sm disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
            >
              <img
                src={example.src}
                alt={`Example: ${example.trueClass}`}
                className="w-full aspect-square object-cover group-hover:opacity-90 transition"
                loading="lazy"
              />
              <span className="flex items-center gap-1 px-1.5 py-1 text-[10px] font-semibold text-slate-600">
                <span className={`h-1.5 w-1.5 rounded-full shrink-0 ${meta.dot}`} />
                <span className="truncate">{meta.shortLabel}</span>
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
