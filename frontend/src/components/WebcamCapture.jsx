/**
 * WebcamCapture — renders a live video feed and exposes a ref + capture button.
 */
export default function WebcamCapture({ videoRef, isCameraActive, onCapture, isLoading }) {
  return (
    <div className="space-y-3">
      <div className="relative rounded-xl overflow-hidden border border-slate-700 bg-black aspect-square">
        {isCameraActive ? (
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="flex items-center justify-center h-full text-slate-500 text-sm">
            Camera not active
          </div>
        )}
      </div>

      {isCameraActive && (
        <button
          type="button"
          onClick={onCapture}
          disabled={isLoading}
          className="w-full py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold text-sm transition"
        >
          {isLoading ? "Analyzing…" : "Capture & Analyze"}
        </button>
      )}
    </div>
  );
}
