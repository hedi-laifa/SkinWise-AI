import React, { useState, useRef, useEffect } from "react";

const API_BASE_URL =
  (typeof import.meta !== "undefined" &&
    import.meta.env &&
    import.meta.env.VITE_API_BASE_URL) ||
  "http://localhost:8000";

const CLASS_ORDER = [
  "Mild (Level 0)",
  "Moderate (Level 1)",
  "Severe (Level 2)",
  "Very Severe (Level 3)",
];

const SEVERITY_STYLES = {
  "Mild (Level 0)": {
    bar: "bg-emerald-600",
    badge: "bg-emerald-50 text-emerald-800 border-emerald-200",
    dot: "bg-emerald-500",
    description: "Grade 0 — Minimal comedones or sparse inflammatory lesions.",
  },
  "Moderate (Level 1)": {
    bar: "bg-amber-500",
    badge: "bg-amber-50 text-amber-800 border-amber-200",
    dot: "bg-amber-500",
    description: "Grade 1 — Moderate papules and pustules localized to facial zones.",
  },
  "Severe (Level 2)": {
    bar: "bg-orange-600",
    badge: "bg-orange-50 text-orange-800 border-orange-200",
    dot: "bg-orange-500",
    description: "Grade 2 — Extensive inflammatory papules, pustules, and early nodules.",
  },
  "Very Severe (Level 3)": {
    bar: "bg-rose-600",
    badge: "bg-rose-50 text-rose-800 border-rose-200",
    dot: "bg-rose-600",
    description: "Grade 3 — High lesion density with deep nodulocystic involvement.",
  },
};

export default function App() {
  const [inputMode, setInputMode] = useState("upload"); // "upload" | "webcam"
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [heatmapUrl, setHeatmapUrl] = useState(null);

  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const mediaStreamRef = useRef(null);

  // Stop active webcam tracks on unmount
  useEffect(() => {
    return () => {
      stopCameraStream();
    };
  }, []);

  const stopCameraStream = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsCameraActive(false);
  };

  const startCameraStream = async () => {
    setErrorMessage(null);
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setErrorMessage(
        "Browser MediaDevices API is not available. Ensure you are accessing via localhost or HTTPS."
      );
      return;
    }

    try {
      stopCameraStream();
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 640 },
          facingMode: "user",
        },
        audio: false,
      });

      mediaStreamRef.current = stream;
      setIsCameraActive(true);

      // Attach stream to video element on next tick
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(() => {});
        }
      }, 50);
    } catch (err) {
      setIsCameraActive(false);
      setErrorMessage(
        err instanceof Error
          ? `Unable to access webcam: ${err.message}`
          : "Camera permission denied or no webcam device detected."
      );
    }
  };

  const handleSwitchMode = (mode) => {
    setInputMode(mode);
    setErrorMessage(null);
    if (mode === "webcam") {
      startCameraStream();
    } else {
      stopCameraStream();
    }
  };

  const validateAndSetFile = (file) => {
    if (!file) return;

    const validTypes = ["image/jpeg", "image/jpg", "image/png"];
    const validExt = /\.(jpe?g|png)$/i.test(file.name || "");

    if (!validTypes.includes(file.type) && !validExt) {
      setErrorMessage("Invalid file format. Please upload a JPEG (.jpg) or PNG (.png) image.");
      return;
    }

    setErrorMessage(null);
    setPrediction(null);
    if (heatmapUrl) {
      URL.revokeObjectURL(heatmapUrl);
      setHeatmapUrl(null);
    }
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  };

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];
    if (file) {
      validateAndSetFile(file);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (!isDragging) setIsDragging(true);
  };

  const handleDragLeave = (event) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);

    const droppedFile = event.dataTransfer.files?.[0];
    if (droppedFile) {
      validateAndSetFile(droppedFile);
    }
  };

  const handleReset = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    if (heatmapUrl) URL.revokeObjectURL(heatmapUrl);
    setSelectedFile(null);
    setPreviewUrl(null);
    setPrediction(null);
    setHeatmapUrl(null);
    setErrorMessage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const executeInferenceForFile = async (fileToAnalyze) => {
    if (!fileToAnalyze) {
      setErrorMessage("Please select or capture a dermatological image before running inference.");
      return;
    }

    setIsLoading(true);
    setErrorMessage(null);

    try {
      const formData = new FormData();
      formData.append("file", fileToAnalyze);

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let detail = `Server responded with HTTP ${response.status}`;
        try {
          const errJson = await response.json();
          if (errJson?.detail) detail = errJson.detail;
        } catch {
          // Ignore JSON parse errors on non-JSON error responses
        }
        throw new Error(detail);
      }

      const data = await response.json();
      setPrediction(data);

      // Fetch Mathematical Grad-CAM heatmap overlay from POST /predict/visualize
      try {
        const vizFormData = new FormData();
        vizFormData.append("file", fileToAnalyze);
        const vizResponse = await fetch(`${API_BASE_URL}/predict/visualize`, {
          method: "POST",
          body: vizFormData,
        });
        if (vizResponse.ok) {
          const blob = await vizResponse.blob();
          setHeatmapUrl((prev) => {
            if (prev) URL.revokeObjectURL(prev);
            return URL.createObjectURL(blob);
          });
        }
      } catch {
        // Keep prediction even if visualization endpoint fails
      }
    } catch (error) {
      setErrorMessage(
        error instanceof Error
          ? error.message
          : "Unable to reach the SkinWise AI inference server at http://localhost:8000."
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleRunInference = async () => {
    await executeInferenceForFile(selectedFile);
  };

  const handleCaptureAndAnalyze = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas || !isCameraActive) {
      setErrorMessage("Webcam stream is not active. Please start the camera first.");
      return;
    }

    const width = video.videoWidth || 640;
    const height = video.videoHeight || 480;
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      setErrorMessage("Unable to initialize 2D canvas context for frame capture.");
      return;
    }

    // Draw current video frame onto canvas
    ctx.drawImage(video, 0, 0, width, height);

    canvas.toBlob(
      async (blob) => {
        if (!blob) {
          setErrorMessage("Failed to encode webcam frame as JPEG blob.");
          return;
        }

        const capturedFile = new File([blob], `webcam_capture_${Date.now()}.jpg`, {
          type: "image/jpeg",
        });

        if (previewUrl) URL.revokeObjectURL(previewUrl);
        setSelectedFile(capturedFile);
        setPreviewUrl(URL.createObjectURL(capturedFile));

        await executeInferenceForFile(capturedFile);
      },
      "image/jpeg",
      0.95
    );
  };

  // Build ordered 4-class entries from prediction.probabilities
  const probabilityEntries = prediction?.probabilities
    ? [
        ...CLASS_ORDER.filter((cls) => cls in prediction.probabilities).map((cls) => [
          cls,
          prediction.probabilities[cls],
        ]),
        ...Object.entries(prediction.probabilities).filter(
          ([cls]) => !CLASS_ORDER.includes(cls)
        ),
      ]
    : [];

  const activeSeverityStyle =
    (prediction && SEVERITY_STYLES[prediction.predicted_class]) ||
    SEVERITY_STYLES["Mild (Level 0)"];

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans antialiased flex flex-col">
      {/* Hidden Canvas for Webcam Frame Extraction */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Academic Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 py-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-lg bg-slate-900 text-white flex items-center justify-center font-bold text-lg tracking-tight shadow-sm">
              SW
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-semibold tracking-tight text-slate-900">
                  SkinWise AI
                </h1>
                <span className="text-xs font-mono uppercase px-2 py-0.5 rounded bg-slate-100 text-slate-700 border border-slate-200">
                  ACNE04 Benchmark
                </span>
              </div>
              <p className="text-xs text-slate-500">
                Clinical Computer Vision Grading &bull; EfficientNet-B0 ONNX Runtime &amp; Grad-CAM
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4 text-xs text-slate-600 font-mono">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-100 border border-slate-200">
              <span className="h-2 w-2 rounded-full bg-emerald-500"></span>
              Input: 224&times;224 RGB
            </span>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-100 border border-slate-200">
              Classes: 4 (Hayashi Scale)
            </span>
          </div>
        </div>
      </header>

      {/* Main Content Grid */}
      <main className="flex-1 max-w-6xl w-full mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left Column: Image Upload + Live Webcam Capture (5 cols) */}
          <section className="lg:col-span-5 bg-white rounded-xl border border-slate-200 shadow-sm p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-base font-semibold text-slate-900">
                  1. Dermatological Image Acquisition
                </h2>
                <p className="text-xs text-slate-500 mt-0.5">
                  Upload a JPEG/PNG specimen or capture a live webcam frame.
                </p>
              </div>
              {(selectedFile || prediction) && (
                <button
                  type="button"
                  onClick={handleReset}
                  className="text-xs font-medium text-slate-500 hover:text-slate-800 underline cursor-pointer"
                >
                  Clear
                </button>
              )}
            </div>

            {/* Mode Switcher Tabs: Drag-and-Drop Upload vs. Live Webcam */}
            <div className="grid grid-cols-2 gap-1.5 p-1 bg-slate-100 rounded-lg border border-slate-200 mb-4">
              <button
                type="button"
                onClick={() => handleSwitchMode("upload")}
                className={`py-2 px-3 rounded-md text-xs font-semibold transition-all cursor-pointer ${
                  inputMode === "upload"
                    ? "bg-white text-slate-900 shadow-xs"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                Drag &amp; Drop Upload
              </button>
              <button
                type="button"
                onClick={() => handleSwitchMode("webcam")}
                className={`py-2 px-3 rounded-md text-xs font-semibold transition-all flex items-center justify-center gap-1.5 cursor-pointer ${
                  inputMode === "webcam"
                    ? "bg-white text-slate-900 shadow-xs"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                <span
                  className={`h-2 w-2 rounded-full ${
                    isCameraActive ? "bg-rose-500 animate-pulse" : "bg-slate-400"
                  }`}
                />
                Live Webcam Feed
              </button>
            </div>

            {inputMode === "upload" ? (
              <>
                {/* Drag and Drop Zone */}
                <div
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      fileInputRef.current?.click();
                    }
                  }}
                  className={`relative group cursor-pointer rounded-xl border-2 border-dashed transition-all duration-150 p-6 text-center ${
                    isDragging
                      ? "border-slate-900 bg-slate-100/70"
                      : selectedFile
                      ? "border-slate-300 bg-slate-50/50 hover:border-slate-400"
                      : "border-slate-300 bg-slate-50 hover:bg-slate-100/60 hover:border-slate-400"
                  }`}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/jpeg,image/png,.jpg,.jpeg,.png"
                    onChange={handleFileChange}
                    className="hidden"
                  />

                  {previewUrl ? (
                    <div className="space-y-3">
                      <div className="relative mx-auto w-56 h-56 rounded-lg overflow-hidden border border-slate-200 bg-slate-900 shadow-inner">
                        <img
                          src={previewUrl}
                          alt="Uploaded skin specimen preview"
                          className="w-full h-full object-cover"
                        />
                      </div>
                      <div className="text-xs text-slate-600 font-mono truncate max-w-xs mx-auto">
                        {selectedFile?.name}{" "}
                        <span className="text-slate-400">
                          ({((selectedFile?.size || 0) / 1024).toFixed(1)} KB)
                        </span>
                      </div>
                      <p className="text-xs text-slate-500">
                        Click or drag another file to replace image
                      </p>
                    </div>
                  ) : (
                    <div className="py-8 space-y-3">
                      <div className="mx-auto h-12 w-12 rounded-full bg-slate-200/70 text-slate-700 flex items-center justify-center">
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          className="h-6 w-6"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                          strokeWidth={1.75}
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
                          />
                        </svg>
                      </div>
                      <div>
                        <p className="text-sm font-medium text-slate-800">
                          Drag &amp; drop a skin image here, or{" "}
                          <span className="underline text-slate-900 font-semibold">
                            browse
                          </span>
                        </p>
                        <p className="text-xs text-slate-500 mt-1">
                          Supports JPEG (.jpg) and PNG (.png) &bull; Preprocessed to 224&times;224
                        </p>
                      </div>
                    </div>
                  )}
                </div>

                {/* Run Inference Action Button */}
                <button
                  type="button"
                  onClick={handleRunInference}
                  disabled={!selectedFile || isLoading}
                  className={`mt-5 w-full py-3 px-4 rounded-lg font-medium text-sm flex items-center justify-center gap-2.5 transition-all shadow-sm ${
                    !selectedFile || isLoading
                      ? "bg-slate-200 text-slate-500 cursor-not-allowed"
                      : "bg-slate-900 text-white hover:bg-slate-800 active:bg-slate-950 cursor-pointer"
                  }`}
                >
                  {isLoading ? (
                    <span>Running ONNX Inference...</span>
                  ) : (
                    <span>Run Inference</span>
                  )}
                </button>
              </>
            ) : (
              /* Live Webcam Capture Panel */
              <div className="space-y-4">
                <div className="relative rounded-xl overflow-hidden border border-slate-300 bg-slate-950 aspect-square flex items-center justify-center">
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className={`w-full h-full object-cover ${
                      isCameraActive ? "block" : "hidden"
                    }`}
                  />
                  {!isCameraActive && (
                    <div className="text-center p-6 space-y-3">
                      <p className="text-xs text-slate-400">
                        Webcam stream is currently paused.
                      </p>
                      <button
                        type="button"
                        onClick={startCameraStream}
                        className="px-3.5 py-2 rounded-lg bg-white text-slate-900 text-xs font-semibold hover:bg-slate-100 cursor-pointer"
                      >
                        Start Camera
                      </button>
                    </div>
                  )}

                  {isCameraActive && (
                    <div className="absolute top-3 left-3 inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-black/65 text-white text-[11px] font-mono backdrop-blur-xs">
                      <span className="h-2 w-2 rounded-full bg-rose-500 animate-pulse" />
                      LIVE WEBCAM
                    </div>
                  )}
                </div>

                <div className="flex gap-2.5">
                  <button
                    type="button"
                    onClick={handleCaptureAndAnalyze}
                    disabled={!isCameraActive || isLoading}
                    className={`flex-1 py-3 px-4 rounded-lg font-medium text-sm flex items-center justify-center gap-2 transition-all shadow-sm ${
                      !isCameraActive || isLoading
                        ? "bg-slate-200 text-slate-500 cursor-not-allowed"
                        : "bg-slate-900 text-white hover:bg-slate-800 active:bg-slate-950 cursor-pointer"
                    }`}
                  >
                    {isLoading ? (
                      <span>Analyzing Frame...</span>
                    ) : (
                      <span>Capture &amp; Analyze</span>
                    )}
                  </button>

                  {isCameraActive ? (
                    <button
                      type="button"
                      onClick={stopCameraStream}
                      className="px-3.5 py-3 rounded-lg border border-slate-300 text-slate-700 hover:bg-slate-100 text-xs font-semibold cursor-pointer"
                    >
                      Stop
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={startCameraStream}
                      className="px-3.5 py-3 rounded-lg border border-slate-300 text-slate-700 hover:bg-slate-100 text-xs font-semibold cursor-pointer"
                    >
                      Start
                    </button>
                  )}
                </div>
              </div>
            )}

            {/* Error Alert */}
            {errorMessage && (
              <div
                role="alert"
                className="mt-4 rounded-lg bg-rose-50 border border-rose-200 px-4 py-3 text-xs text-rose-800 flex items-start gap-2.5"
              >
                <span className="font-bold uppercase tracking-wider text-rose-600 shrink-0">
                  Error:
                </span>
                <span>{errorMessage}</span>
              </div>
            )}
          </section>

          {/* Right Column: Quantitative Diagnostic Output & Grad-CAM (7 cols) */}
          <section className="lg:col-span-7 bg-white rounded-xl border border-slate-200 shadow-sm p-6">
            <div className="flex flex-wrap items-center justify-between gap-2 pb-4 border-b border-slate-100">
              <div>
                <h2 className="text-base font-semibold text-slate-900">
                  2. Inference Results &amp; Grad-CAM Explainability
                </h2>
                <p className="text-xs text-slate-500 mt-0.5">
                  Posterior softmax distribution and convolutional activation mapping.
                </p>
              </div>

              {/* Execution Latency Telemetry Pill */}
              {prediction && typeof prediction.latency_ms === "number" && (
                <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 text-white text-xs font-mono">
                  <span className="text-slate-400">ONNX Latency:</span>
                  <span className="font-semibold text-emerald-400">
                    {prediction.latency_ms.toFixed(2)} ms
                  </span>
                </div>
              )}
            </div>

            {!prediction && !isLoading && (
              <div className="py-16 text-center">
                <p className="text-sm font-medium text-slate-600">
                  No active inference session
                </p>
                <p className="text-xs text-slate-400 mt-1 max-w-md mx-auto">
                  Upload a JPEG/PNG image or click{" "}
                  <span className="font-semibold text-slate-600">
                    Capture &amp; Analyze
                  </span>{" "}
                  from the live webcam feed to compute ACNE04 severity probabilities and Grad-CAM heatmaps.
                </p>
              </div>
            )}

            {isLoading && (
              <div className="py-16 text-center space-y-3">
                <div className="inline-block h-8 w-8 animate-spin rounded-full border-2 border-slate-900 border-t-transparent" />
                <p className="text-sm font-medium text-slate-700">
                  Executing EfficientNet-B0 ONNX Runtime inference &amp; Grad-CAM...
                </p>
                <p className="text-xs text-slate-400 font-mono">
                  POST {API_BASE_URL}/predict
                </p>
              </div>
            )}

            {prediction && !isLoading && (
              <div className="mt-5 space-y-6">
                {/* Prominent Predicted Class Banner */}
                <div
                  className={`rounded-xl border p-5 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 ${activeSeverityStyle.badge}`}
                >
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span
                        className={`h-2.5 w-2.5 rounded-full ${activeSeverityStyle.dot}`}
                      />
                      <span className="text-xs font-mono uppercase tracking-wider opacity-80">
                        Predicted ACNE04 Severity Class
                      </span>
                    </div>
                    <h3 className="text-2xl font-bold tracking-tight">
                      {prediction.predicted_class}
                    </h3>
                    <p className="text-xs opacity-85">
                      {activeSeverityStyle.description}
                    </p>
                  </div>

                  <div className="sm:text-right shrink-0 bg-white/80 backdrop-blur px-4 py-2.5 rounded-lg border border-black/5">
                    <span className="block text-[11px] font-mono uppercase text-slate-500">
                      Top-1 Confidence
                    </span>
                    <span className="text-xl font-bold font-mono text-slate-900">
                      {(
                        (prediction.confidence ??
                          prediction.probabilities?.[prediction.predicted_class] ??
                          0) * 100
                      ).toFixed(2)}
                      %
                    </span>
                  </div>
                </div>

                {/* 4 Distinct Horizontal Progress Bars */}
                <div className="space-y-4">
                  <div className="flex items-center justify-between text-xs font-mono uppercase tracking-wider text-slate-500">
                    <span>Severity Class Distribution (4 Classes)</span>
                    <span>Softmax Probability</span>
                  </div>

                  <div className="space-y-3.5">
                    {probabilityEntries.map(([className, probValue]) => {
                      const pct = Math.max(
                        0,
                        Math.min(100, Number(probValue) * 100)
                      );
                      const isWinner = className === prediction.predicted_class;
                      const style =
                        SEVERITY_STYLES[className] ||
                        SEVERITY_STYLES["Mild (Level 0)"];

                      return (
                        <div
                          key={className}
                          className={`p-3.5 rounded-lg border transition-colors ${
                            isWinner
                              ? "border-slate-900 bg-slate-50/90 shadow-xs"
                              : "border-slate-200 bg-white"
                          }`}
                        >
                          <div className="flex items-center justify-between text-sm mb-2">
                            <div className="flex items-center gap-2">
                              <span
                                className={`h-2.5 w-2.5 rounded-full ${style.dot}`}
                              />
                              <span
                                className={
                                  isWinner
                                    ? "font-semibold text-slate-900"
                                    : "font-medium text-slate-700"
                                }
                              >
                                {className}
                              </span>
                              {isWinner && (
                                <span className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded bg-slate-900 text-white">
                                  Argmax
                                </span>
                              )}
                            </div>
                            <span className="font-mono text-xs font-semibold text-slate-800">
                              {pct.toFixed(2)}%
                            </span>
                          </div>

                          <div className="w-full h-3 bg-slate-100 rounded-full overflow-hidden border border-slate-200/60">
                            <div
                              className={`h-full rounded-full transition-all duration-500 ease-out ${style.bar}`}
                              style={{ width: `${pct}%` }}
                              role="progressbar"
                              aria-valuenow={Number(pct.toFixed(2))}
                              aria-valuemin={0}
                              aria-valuemax={100}
                              aria-label={`${className} probability`}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Side-by-Side Original Specimen & Mathematical Grad-CAM Heatmap Overlay */}
                {(previewUrl || heatmapUrl) && (
                  <div className="pt-4 border-t border-slate-100 space-y-3">
                    <div className="flex items-center justify-between">
                      <h4 className="text-xs font-mono uppercase tracking-wider text-slate-600">
                        Visual Explainability (Original RGB vs. Grad-CAM Activation Map)
                      </h4>
                      <span className="text-[11px] font-mono text-slate-400">
                        POST /predict/visualize
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      {previewUrl && (
                        <div className="space-y-1.5">
                          <div className="aspect-square w-full max-w-[200px] rounded-lg overflow-hidden border border-slate-200 bg-slate-900">
                            <img
                              src={previewUrl}
                              alt="Original RGB input"
                              className="w-full h-full object-cover"
                            />
                          </div>
                          <p className="text-[11px] font-mono text-slate-500">
                            Input Specimen (RGB)
                          </p>
                        </div>
                      )}

                      {heatmapUrl && (
                        <div className="space-y-1.5">
                          <div className="aspect-square w-full max-w-[200px] rounded-lg overflow-hidden border border-slate-200 bg-slate-900">
                            <img
                              src={heatmapUrl}
                              alt="Grad-CAM heatmap overlay"
                              className="w-full h-full object-cover"
                            />
                          </div>
                          <p className="text-[11px] font-mono text-slate-500">
                            Grad-CAM Overlay (224&times;224)
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </section>
        </div>
      </main>

      {/* Academic Footer Disclaimer */}
      <footer className="border-t border-slate-200 bg-white py-4 px-6 text-center text-xs text-slate-500">
        SkinWise AI &mdash; Academic MLOps &amp; Computer Vision Research Prototype (ACNE04 Dataset). Not intended for standalone clinical diagnosis.
      </footer>
    </div>
  );
}
