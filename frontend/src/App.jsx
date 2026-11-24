/**
 * SkinWise AI — main page.
 * Holds the application state and the calls to the FastAPI backend;
 * all visual pieces live in ./components.
 */
import { useState, useRef, useEffect } from "react";
import { checkHealth, predict, visualize } from "./services/api.js";
import { CLASS_ORDER, SEVERITY_META } from "./constants/severity.js";

import Header from "./components/Header.jsx";
import SeverityCard from "./components/SeverityCard.jsx";
import Dropzone from "./components/Dropzone.jsx";
import WebcamCapture from "./components/WebcamCapture.jsx";
import ScanHistory from "./components/ScanHistory.jsx";
import ExampleGallery from "./components/ExampleGallery.jsx";
import { EmptyState, LoadingState } from "./components/AnalysisStates.jsx";
import PredictionBanner from "./components/PredictionBanner.jsx";
import GradCamViewer from "./components/GradCamViewer.jsx";
import ProbabilityBars from "./components/ProbabilityBars.jsx";

export default function App() {
  const [inputMode, setInputMode] = useState("upload");
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [imageDimensions, setImageDimensions] = useState(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [heatmapUrl, setHeatmapUrl] = useState(null);
  const [apiHealth, setApiHealth] = useState({
    checking: true,
    online: false,
    modelLoaded: false,
  });
  const [scanHistory, setScanHistory] = useState([]);

  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const mediaStreamRef = useRef(null);

  // ── Backend health ───────────────────────────────────────────────────
  const checkBackendHealth = async () => {
    setApiHealth((prev) => ({ ...prev, checking: true }));
    try {
      const data = await checkHealth();
      setApiHealth({ checking: false, ...data });
    } catch {
      setApiHealth({ checking: false, online: false, modelLoaded: false });
    }
  };

  useEffect(() => {
    checkBackendHealth();
    return () => stopCameraStream();
  }, []);

  // ── Webcam ───────────────────────────────────────────────────────────
  const stopCameraStream = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    setIsCameraActive(false);
  };

  const startCameraStream = async () => {
    setErrorMessage(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setErrorMessage("Browser MediaDevices API unavailable. Use localhost or HTTPS.");
      return;
    }
    try {
      stopCameraStream();
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 640 }, facingMode: "user" },
        audio: false,
      });
      mediaStreamRef.current = stream;
      setIsCameraActive(true);
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(() => {});
        }
      }, 50);
    } catch (err) {
      setIsCameraActive(false);
      setErrorMessage(
        err instanceof Error ? `Unable to access webcam: ${err.message}` : "Camera permission denied."
      );
    }
  };

  const handleSwitchMode = (mode) => {
    setInputMode(mode);
    setErrorMessage(null);
    if (mode === "webcam") startCameraStream();
    else stopCameraStream();
  };

  // ── File selection ───────────────────────────────────────────────────
  const validateAndSetFile = (file) => {
    if (!file) return;
    const validTypes = ["image/jpeg", "image/jpg", "image/png"];
    if (!validTypes.includes(file.type) && !/\.(jpe?g|png)$/i.test(file.name)) {
      setErrorMessage("Invalid format. Upload a JPEG or PNG image.");
      return null;
    }
    setErrorMessage(null);
    setPrediction(null);
    setHeatmapUrl(null);
    setImageDimensions(null);
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => setImageDimensions({ width: img.naturalWidth, height: img.naturalHeight });
    img.src = url;
    setSelectedFile(file);
    setPreviewUrl(url);
    return url;
  };

  const handleReset = () => {
    setSelectedFile(null); setPreviewUrl(null); setImageDimensions(null);
    setPrediction(null); setHeatmapUrl(null); setErrorMessage(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  // ── Inference ────────────────────────────────────────────────────────
  const executeInference = async (file, preview) => {
    if (!file) { setErrorMessage("Select or capture an image before running inference."); return; }
    setIsLoading(true); setErrorMessage(null);
    try {
      // Client-side latency: full round trip seen by the browser
      // (upload + server preprocessing + ONNX inference + JSON response).
      const t0 = performance.now();
      const result = await predict(file);
      const data = { ...result, client_latency_ms: performance.now() - t0 };
      setPrediction(data);
      setApiHealth({ checking: false, online: true, modelLoaded: true });
      let cam = null;
      try {
        cam = await visualize(file);
        setHeatmapUrl(cam);
      } catch {
        // Grad-CAM is optional: keep showing the prediction even if the heatmap request fails
      }
      setScanHistory((prev) => [{
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        fileName: file.name,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
        previewUrl: preview || previewUrl,
        heatmapUrl: cam,
        prediction: data,
      }, ...prev].slice(0, 5));
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Unable to reach the inference server at http://localhost:8000.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleRunInference = () => executeInference(selectedFile, previewUrl);

  const handleExampleSelect = async (example) => {
    setErrorMessage(null);
    try {
      const res = await fetch(example.src);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      const file = new File([blob], example.fileName, { type: blob.type || "image/jpeg" });
      const url = validateAndSetFile(file);
      if (url) await executeInference(file, url);
    } catch (err) {
      setErrorMessage(`Unable to load the example image: ${err instanceof Error ? err.message : err}`);
    }
  };

  const handleCaptureAndAnalyze = () => {
    const video = videoRef.current; const canvas = canvasRef.current;
    if (!video || !canvas || !isCameraActive) { setErrorMessage("Webcam stream is not active."); return; }
    const w = video.videoWidth || 640; const h = video.videoHeight || 480;
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) { setErrorMessage("Unable to initialize canvas context."); return; }
    ctx.drawImage(video, 0, 0, w, h);
    canvas.toBlob(async (blob) => {
      if (!blob) { setErrorMessage("Failed to encode webcam frame."); return; }
      const f = new File([blob], `webcam_${Date.now()}.jpg`, { type: "image/jpeg" });
      const url = URL.createObjectURL(f);
      setSelectedFile(f); setPreviewUrl(url); setImageDimensions({ width: w, height: h });
      await executeInference(f, url);
    }, "image/jpeg", 0.95);
  };

  const handleHistorySelect = (item) => {
    setPreviewUrl(item.previewUrl);
    setHeatmapUrl(item.heatmapUrl);
    setPrediction(item.prediction);
  };

  // ── Layout ───────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 flex flex-col">
      <canvas ref={canvasRef} className="hidden" />

      <Header apiHealth={apiHealth} onRetry={checkBackendHealth} />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-5">

        {/* Severity Reference Strip */}
        <section className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          {CLASS_ORDER.map((cls) => (
            <SeverityCard
              key={cls}
              meta={SEVERITY_META[cls]}
              isPredicted={prediction?.predicted_class === cls}
            />
          ))}
        </section>

        {/* Two-column grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">

          {/* ── Left: Image Input (5 cols) ─────────────────────────────── */}
          <section className="lg:col-span-5 bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-[11px] font-semibold uppercase tracking-widest text-blue-600 mb-0.5">Step 1 · Input</p>
                <h2 className="text-sm font-bold text-slate-900">Image Acquisition</h2>
              </div>
              {(selectedFile || prediction) && (
                <button
                  type="button"
                  onClick={handleReset}
                  className="text-xs font-medium text-slate-400 hover:text-red-500 transition px-2 py-1 rounded hover:bg-red-50 border border-transparent hover:border-red-100"
                >
                  Reset
                </button>
              )}
            </div>

            {/* Mode Switcher */}
            <div className="flex p-1 gap-1 bg-slate-100 rounded-lg border border-slate-200">
              {["upload", "webcam"].map((mode) => (
                <button
                  key={mode}
                  type="button"
                  onClick={() => handleSwitchMode(mode)}
                  className={`flex-1 py-2 text-xs font-semibold rounded-md transition ${
                    inputMode === mode
                      ? "bg-white text-blue-700 shadow-sm border border-slate-200"
                      : "text-slate-500 hover:text-slate-700"
                  }`}
                >
                  {mode === "upload" ? "Upload Image" : "Live Camera"}
                </button>
              ))}
            </div>

            {inputMode === "upload" ? (
              <Dropzone
                fileInputRef={fileInputRef}
                selectedFile={selectedFile}
                previewUrl={previewUrl}
                imageDimensions={imageDimensions}
                isLoading={isLoading}
                onFileSelected={validateAndSetFile}
                onAnalyze={handleRunInference}
              />
            ) : (
              <WebcamCapture
                videoRef={videoRef}
                isCameraActive={isCameraActive}
                isLoading={isLoading}
                onStart={startCameraStream}
                onStop={stopCameraStream}
                onCapture={handleCaptureAndAnalyze}
              />
            )}

            {inputMode === "upload" && (
              <ExampleGallery onSelect={handleExampleSelect} isLoading={isLoading} />
            )}

            {/* Error */}
            {errorMessage && (
              <div role="alert" className="rounded-lg bg-red-50 border border-red-200 px-3 py-2.5 text-xs text-red-700 flex items-start gap-2">
                <svg className="w-3.5 h-3.5 text-red-500 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                <span>{errorMessage}</span>
              </div>
            )}

            <ScanHistory items={scanHistory} onSelect={handleHistorySelect} />
          </section>

          {/* ── Right: Diagnostic Report (7 cols) ──────────────────────── */}
          <section className="lg:col-span-7 bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-5">
            <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-slate-100">
              <div>
                <p className="text-[11px] font-semibold uppercase tracking-widest text-blue-600 mb-0.5">Step 2 · Analysis</p>
                <h2 className="text-sm font-bold text-slate-900">Diagnostic Report & Grad-CAM</h2>
              </div>
              {prediction?.latency_ms != null && (
                <div className="flex flex-wrap items-center gap-1.5">
                  <span
                    title="Server side: ONNX Runtime inference time, measured by FastAPI"
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-50 border border-slate-200 text-xs font-mono text-slate-500"
                  >
                    <span className="h-1.5 w-1.5 rounded-full bg-green-500" />
                    Server (ONNX): {prediction.latency_ms.toFixed(1)} ms
                  </span>
                  {prediction.client_latency_ms != null && (
                    <span
                      title="Client side: full round trip measured in the browser (upload + preprocessing + inference + response)"
                      className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-50 border border-slate-200 text-xs font-mono text-slate-500"
                    >
                      <span className="h-1.5 w-1.5 rounded-full bg-blue-500" />
                      Client (round trip): {prediction.client_latency_ms.toFixed(0)} ms
                    </span>
                  )}
                </div>
              )}
            </div>

            {!prediction && !isLoading && <EmptyState />}
            {isLoading && <LoadingState />}

            {prediction && !isLoading && (
              <div className="space-y-5">
                <PredictionBanner prediction={prediction} />
                <GradCamViewer
                  previewUrl={previewUrl}
                  heatmapUrl={heatmapUrl}
                  predictedClass={prediction.predicted_class}
                />
                <ProbabilityBars
                  probabilities={prediction.probabilities}
                  predictedClass={prediction.predicted_class}
                />
              </div>
            )}
          </section>
        </div>
      </main>

      {/* ── Footer ───────────────────────────────────────────────────────── */}
      <footer className="border-t border-slate-200 bg-white py-3 px-6 text-center text-[11px] text-slate-400">
        SkinWise AI — EfficientNet-B0 · ACNE04 · Hayashi Grading · Research prototype, not a substitute for professional dermatological advice.
      </footer>
    </div>
  );
}
