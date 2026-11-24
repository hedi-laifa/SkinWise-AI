/**
 * GradCamViewer — original image vs. Grad-CAM heatmap, side by side or blended with an opacity slider.
 */
import { useState } from "react";

export default function GradCamViewer({ previewUrl, heatmapUrl, predictedClass }) {
  const [vizMode, setVizMode] = useState("split");
  const [blendOpacity, setBlendOpacity] = useState(65);

  if (!previewUrl && !heatmapUrl) return null;

  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="text-[11px] font-semibold text-blue-700 uppercase tracking-widest">Grad-CAM Activation</p>
          <p className="text-[11px] text-slate-500 mt-0.5">Regions driving prediction for <span className="font-medium text-slate-700">{predictedClass}</span></p>
        </div>
        {previewUrl && heatmapUrl && (
          <div className="inline-flex p-0.5 gap-0.5 bg-slate-200 rounded-lg text-xs">
            {["split", "blend"].map((m) => (
              <button key={m} type="button" onClick={() => setVizMode(m)}
                className={`px-2.5 py-1 rounded-md font-medium transition cursor-pointer ${
                  vizMode === m ? "bg-white text-blue-700 shadow-sm" : "text-slate-500 hover:text-slate-700"
                }`}>
                {m === "split" ? "Side-by-Side" : "Blend"}
              </button>
            ))}
          </div>
        )}
      </div>

      {vizMode === "split" || !heatmapUrl ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {previewUrl && (
            <div className="relative aspect-square rounded-lg overflow-hidden border border-slate-200 bg-white shadow-sm">
              <img src={previewUrl} alt="Original" className="w-full h-full object-cover" />
              <span className="absolute top-2 left-2 px-2 py-0.5 rounded text-[10px] font-mono bg-white/90 text-slate-600 border border-slate-200">
                Original
              </span>
            </div>
          )}
          {heatmapUrl && (
            <div className="relative aspect-square rounded-lg overflow-hidden border border-blue-200 bg-white shadow-sm">
              <img src={heatmapUrl} alt="Grad-CAM" className="w-full h-full object-cover" />
              <span className="absolute top-2 left-2 px-2 py-0.5 rounded text-[10px] font-mono bg-white/90 text-blue-700 border border-blue-200">
                Grad-CAM
              </span>
            </div>
          )}
        </div>
      ) : (
        <div className="space-y-3">
          <div className="relative mx-auto max-w-xs aspect-square rounded-lg overflow-hidden border border-blue-200 bg-white shadow-sm">
            <img src={previewUrl} alt="Base" className="w-full h-full object-cover" />
            <img src={heatmapUrl} alt="Overlay" style={{ opacity: blendOpacity / 100 }}
              className="absolute inset-0 w-full h-full object-cover transition-opacity duration-75" />
            <span className="absolute top-2 left-2 px-2 py-0.5 rounded text-[10px] font-mono bg-white/90 text-slate-600 border border-slate-200">
              Opacity: {blendOpacity}%
            </span>
          </div>
          <div className="max-w-xs mx-auto flex items-center gap-2 text-[11px] font-mono text-slate-400">
            <span>RGB</span>
            <input type="range" min={0} max={100} value={blendOpacity}
              onChange={(e) => setBlendOpacity(Number(e.target.value))}
              className="flex-1 accent-blue-600 cursor-pointer" />
            <span>CAM</span>
          </div>
        </div>
      )}

      {/* Colormap bar */}
      <div className="flex items-center justify-between gap-2 text-[10px] font-mono text-slate-400 pt-1">
        <span>Low activation</span>
        <div className="flex-1 max-w-xs h-1.5 rounded-full bg-gradient-to-r from-blue-600 via-emerald-400 via-yellow-400 to-rose-600 border border-slate-200" />
        <span>High salience</span>
      </div>
    </div>
  );
}
