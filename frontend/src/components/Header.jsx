/**
 * Header — app title and live backend / ONNX engine health badge.
 */
export default function Header({ apiHealth, onRetry }) {
  return (
    <header className="sticky top-0 z-30 bg-white border-b border-slate-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          {/* Logo mark */}
          <div className="h-9 w-9 rounded-lg bg-blue-700 flex items-center justify-center shrink-0">
            <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23-.693L5 14.5m14.8.8l1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0112 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5" />
            </svg>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-bold text-slate-900 tracking-tight">SkinWise AI</h1>
              <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-blue-50 text-blue-700 border border-blue-200">
                ACNE04 · Hayashi Grading
              </span>
            </div>
            <p className="text-[11px] text-slate-500">
              EfficientNet-B0 ONNX · Grad-CAM Explainability · MLflow Tracked
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs">
          <button
            type="button"
            onClick={onRetry}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border font-medium transition ${
              apiHealth.online
                ? "bg-green-50 border-green-200 text-green-700 hover:bg-green-100"
                : "bg-red-50 border-red-200 text-red-600 hover:bg-red-100"
            }`}
          >
            <span className={`h-2 w-2 rounded-full ${
              apiHealth.checking ? "bg-amber-400 animate-ping" :
              apiHealth.online ? "bg-green-500" : "bg-red-500"
            }`} />
            {apiHealth.checking ? "Connecting…" : apiHealth.online ? "ONNX Engine Ready" : "API Offline (Retry)"}
          </button>
          <span className="hidden md:inline-flex items-center gap-1 px-2.5 py-1.5 rounded-full bg-slate-100 border border-slate-200 text-slate-500 text-[11px] font-mono">
            224 × 224 RGB
          </span>
        </div>
      </div>
    </header>
  );
}
