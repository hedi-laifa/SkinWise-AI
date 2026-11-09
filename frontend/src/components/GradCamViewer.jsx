/**
 * GradCamViewer — side-by-side or blended Grad-CAM heatmap overlay viewer.
 */
import { useState } from "react";

export default function GradCamViewer({ originalUrl, heatmapUrl }) {
  const [mode, setMode] = useState("split"); // "split" | "blend"
  const [opacity, setOpacity] = useState(65);

  if (!heatmapUrl) return null;

  return (
    <div className="space-y-3">
      {/* Controls */}
      <div className="flex items-center gap-2 flex-wrap">
        {["split", "blend"].map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => setMode(m)}
            className={`text-xs px-3 py-1 rounded-full border font-medium transition ${
              mode === m
                ? "bg-indigo-500/20 border-indigo-500/40 text-indigo-300"
                : "border-slate-700 text-slate-400 hover:text-slate-200"
            }`}
          >
            {m === "split" ? "Side-by-Side" : "Blend"}
          </button>
        ))}
        {mode === "blend" && (
          <input
            type="range"
            min={0}
            max={100}
            value={opacity}
            onChange={(e) => setOpacity(Number(e.target.value))}
            className="flex-1 min-w-[80px] accent-indigo-500"
          />
        )}
      </div>

      {/* Images */}
      {mode === "split" ? (
        <div className="grid grid-cols-2 gap-2">
          <div className="relative rounded-lg overflow-hidden border border-slate-700">
            <img src={originalUrl} alt="Original" className="w-full object-cover" />
            <span className="absolute bottom-1 left-1 text-[10px] bg-black/60 text-white px-1.5 py-0.5 rounded">
              Original
            </span>
          </div>
          <div className="relative rounded-lg overflow-hidden border border-indigo-500/30">
            <img src={heatmapUrl} alt="Grad-CAM" className="w-full object-cover" />
            <span className="absolute bottom-1 left-1 text-[10px] bg-black/60 text-indigo-300 px-1.5 py-0.5 rounded">
              Grad-CAM
            </span>
          </div>
        </div>
      ) : (
        <div className="relative rounded-lg overflow-hidden border border-slate-700">
          <img src={originalUrl} alt="Original" className="w-full object-cover" />
          <img
            src={heatmapUrl}
            alt="Grad-CAM overlay"
            className="absolute inset-0 w-full h-full object-cover"
            style={{ opacity: opacity / 100 }}
          />
        </div>
      )}
    </div>
  );
}
