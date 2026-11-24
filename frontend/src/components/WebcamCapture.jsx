/**
 * WebcamCapture — live camera feed with a framing guide and capture controls.
 * The <video> element stays mounted so `videoRef` keeps its stream.
 */
export default function WebcamCapture({ videoRef, isCameraActive, isLoading, onStart, onStop, onCapture }) {
  return (
    <div className="space-y-3">
      <div className="relative rounded-xl overflow-hidden border border-slate-200 bg-slate-900 aspect-square flex items-center justify-center">
        <video ref={videoRef} autoPlay playsInline muted className={`w-full h-full object-cover ${isCameraActive ? "block" : "hidden"}`} />
        {isCameraActive && (
          <>
            <div className="pointer-events-none absolute inset-8 border border-white/20 rounded-xl">
              <div className="absolute -top-px -left-px w-5 h-5 border-t-2 border-l-2 border-blue-400 rounded-tl" />
              <div className="absolute -top-px -right-px w-5 h-5 border-t-2 border-r-2 border-blue-400 rounded-tr" />
              <div className="absolute -bottom-px -left-px w-5 h-5 border-b-2 border-l-2 border-blue-400 rounded-bl" />
              <div className="absolute -bottom-px -right-px w-5 h-5 border-b-2 border-r-2 border-blue-400 rounded-br" />
            </div>
            <div className="absolute top-2.5 left-2.5 inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-900/80 text-white text-[10px] font-mono border border-slate-700">
              <span className="h-1.5 w-1.5 rounded-full bg-red-500 animate-ping" />
              LIVE
            </div>
          </>
        )}
        {!isCameraActive && (
          <div className="text-center p-6 space-y-3">
            <div className="mx-auto h-10 w-10 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-400">
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
            </div>
            <p className="text-xs text-slate-400">Camera not active</p>
            <button type="button" onClick={onStart} className="px-4 py-1.5 rounded-lg bg-blue-600 text-white text-xs font-semibold hover:bg-blue-500 cursor-pointer">
              Enable Camera
            </button>
          </div>
        )}
      </div>
      <div className="flex gap-2">
        <button
          type="button"
          onClick={onCapture}
          disabled={!isCameraActive || isLoading}
          className={`flex-1 py-2.5 rounded-lg font-semibold text-sm flex items-center justify-center gap-2 transition ${
            !isCameraActive || isLoading ? "bg-slate-100 text-slate-400 cursor-not-allowed border border-slate-200" : "bg-blue-700 text-white hover:bg-blue-800 cursor-pointer"
          }`}
        >
          {isLoading ? "Analyzing…" : "Capture & Analyze"}
        </button>
        <button
          type="button"
          onClick={isCameraActive ? onStop : onStart}
          className="px-4 py-2.5 rounded-lg border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 text-xs font-medium cursor-pointer"
        >
          {isCameraActive ? "Stop" : "Start"}
        </button>
      </div>
    </div>
  );
}
