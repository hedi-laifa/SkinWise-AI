/**
 * Dropzone — drag & drop / click-to-browse image upload with preview,
 * followed by the "Analyze" button.
 */
import { useRef, useState } from "react";

export default function Dropzone({
  selectedFile,
  previewUrl,
  imageDimensions,
  isLoading,
  onFileSelected,
  onAnalyze,
  fileInputRef: externalInputRef,
}) {
  const [isDragging, setIsDragging] = useState(false);
  const internalInputRef = useRef(null);
  const fileInputRef = externalInputRef ?? internalInputRef;

  const handleFileChange = (e) => {
    const f = e.target.files?.[0];
    if (f) onFileSelected(f);
  };

  const handleDragOver = (e) => { e.preventDefault(); e.stopPropagation(); if (!isDragging) setIsDragging(true); };
  const handleDragLeave = (e) => { e.preventDefault(); e.stopPropagation(); setIsDragging(false); };
  const handleDrop = (e) => {
    e.preventDefault(); e.stopPropagation(); setIsDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) onFileSelected(f);
  };

  return (
    <>
      {/* Drop Zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInputRef.current?.click(); } }}
        className={`cursor-pointer rounded-xl border-2 border-dashed transition-all duration-150 p-4 text-center ${
          isDragging
            ? "border-blue-400 bg-blue-50"
            : selectedFile
            ? "border-slate-200 bg-slate-50 hover:border-blue-300"
            : "border-slate-300 bg-slate-50 hover:border-blue-300 hover:bg-blue-50"
        }`}
      >
        <input ref={fileInputRef} type="file" accept="image/jpeg,image/png,.jpg,.jpeg,.png" onChange={handleFileChange} className="hidden" />

        {previewUrl ? (
          <div className="space-y-3">
            <div className="relative mx-auto w-full max-w-[260px] aspect-square rounded-lg overflow-hidden border border-slate-200 bg-white shadow-sm">
              <img src={previewUrl} alt="Preview" className="w-full h-full object-cover" />
              <div className="absolute inset-0 bg-slate-900/0 hover:bg-slate-900/10 transition flex items-end justify-center p-2">
                <span className="text-[10px] font-medium text-slate-600 bg-white/90 px-2 py-1 rounded-full border border-slate-200 opacity-0 hover:opacity-100 transition">
                  Click to replace
                </span>
              </div>
            </div>
            <div className="flex flex-wrap items-center justify-center gap-1.5 text-[11px] font-mono">
              <span className="px-2 py-0.5 rounded bg-slate-100 border border-slate-200 text-slate-600 truncate max-w-[180px]">{selectedFile?.name}</span>
              <span className="px-2 py-0.5 rounded bg-slate-100 border border-slate-200 text-slate-500">{((selectedFile?.size || 0) / 1024).toFixed(1)} KB</span>
              {imageDimensions && (
                <span className="px-2 py-0.5 rounded bg-blue-50 border border-blue-200 text-blue-700">
                  {imageDimensions.width}×{imageDimensions.height}
                </span>
              )}
            </div>
          </div>
        ) : (
          <div className="py-8 space-y-3">
            <div className="mx-auto h-12 w-12 rounded-xl bg-blue-50 border border-blue-100 text-blue-600 flex items-center justify-center">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            </div>
            <div>
              <p className="text-sm font-medium text-slate-700">Drop an image here, or <span className="text-blue-600 underline underline-offset-2">browse</span></p>
              <p className="text-[11px] text-slate-400 mt-0.5">JPEG · PNG · Auto-resized to 224 × 224</p>
            </div>
          </div>
        )}
      </div>

      {/* Analyze Button */}
      <button
        type="button"
        onClick={onAnalyze}
        disabled={!selectedFile || isLoading}
        className={`w-full py-3 rounded-lg font-semibold text-sm flex items-center justify-center gap-2 transition ${
          !selectedFile || isLoading
            ? "bg-slate-100 text-slate-400 border border-slate-200 cursor-not-allowed"
            : "bg-blue-700 text-white hover:bg-blue-800 shadow-sm cursor-pointer"
        }`}
      >
        {isLoading ? (
          <>
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
            Running ONNX Inference & Grad-CAM…
          </>
        ) : (
          <>
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            Analyze Skin Severity
          </>
        )}
      </button>
    </>
  );
}
