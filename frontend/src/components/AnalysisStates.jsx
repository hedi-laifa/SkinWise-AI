/**
 * EmptyState / LoadingState — what the results panel shows before and during inference.
 */

const PIPELINE_STEPS = [
  { step: "01", label: "Preprocess", detail: "Bilinear resize → 224×224, ImageNet normalization" },
  { step: "02", label: "Inference", detail: "EfficientNet-B0 ONNX forward pass, softmax output" },
  { step: "03", label: "Explainability", detail: "Grad-CAM hooks on final convolutional block" },
];

export function EmptyState() {
  return (
    <div className="py-10 text-center space-y-6">
      <div className="inline-flex items-center justify-center w-10 h-10 rounded-xl bg-blue-50 border border-blue-100 text-blue-600">
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      </div>
      <div className="max-w-sm mx-auto">
        <h3 className="text-sm font-semibold text-slate-700 mb-1">Ready for analysis</h3>
        <p className="text-xs text-slate-500 leading-relaxed">
          Upload a facial skin image to compute 4-class Hayashi severity probabilities and Grad-CAM activation maps.
        </p>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-left max-w-lg mx-auto">
        {PIPELINE_STEPS.map(({ step, label, detail }) => (
          <div key={step} className="p-3 rounded-lg bg-slate-50 border border-slate-200">
            <p className="text-[10px] font-semibold text-blue-600 uppercase tracking-widest mb-0.5">{step}</p>
            <p className="text-xs font-semibold text-slate-700">{label}</p>
            <p className="text-[11px] text-slate-500 leading-relaxed mt-0.5">{detail}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export function LoadingState() {
  return (
    <div className="py-16 text-center space-y-3">
      <div className="relative mx-auto w-10 h-10">
        <div className="absolute inset-0 rounded-full border-2 border-slate-200" />
        <div className="absolute inset-0 rounded-full border-2 border-blue-600 border-t-transparent animate-spin" />
      </div>
      <p className="text-sm font-medium text-slate-700">Running inference & Grad-CAM…</p>
      <p className="text-[11px] font-mono text-slate-400">POST /predict · /predict/visualize</p>
    </div>
  );
}
