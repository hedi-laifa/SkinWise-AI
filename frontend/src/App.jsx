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

const SEVERITY_META = {
  "Mild (Level 0)": {
    level: "Grade 0",
    shortLabel: "Mild",
    lesionRange: "1–5 papules/pustules",
    barGradient: "from-emerald-500 to-teal-400",
    ringColor: "#10b981",
    badgeBg: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
    cardBorder: "border-emerald-500/40 bg-emerald-950/20",
    dot: "bg-emerald-400",
    description:
      "Minimal comedones or sparse superficial inflammatory lesions localized to a single facial zone.",
    recommendation:
      "Routine gentle cleansing and topical OTC keratolytics (salicylic acid / benzoyl peroxide).",
  },
  "Moderate (Level 1)": {
    level: "Grade 1",
    shortLabel: "Moderate",
    lesionRange: "6–20 papules/pustules",
    barGradient: "from-amber-500 to-yellow-400",
    ringColor: "#f59e0b",
    badgeBg: "bg-amber-500/15 text-amber-300 border-amber-500/30",
    cardBorder: "border-amber-500/40 bg-amber-950/20",
    dot: "bg-amber-400",
    description:
      "Moderate papules and pustules distributed across multiple facial zones without deep scarring.",
    recommendation:
      "Consider combination topical retinoid + antimicrobial therapy under dermatological guidance.",
  },
  "Severe (Level 2)": {
    level: "Grade 2",
    shortLabel: "Severe",
    lesionRange: "21–50 papules/pustules",
    barGradient: "from-orange-500 to-amber-500",
    ringColor: "#f97316",
    badgeBg: "bg-orange-500/15 text-orange-300 border-orange-500/30",
    cardBorder: "border-orange-500/40 bg-orange-950/20",
    dot: "bg-orange-400",
    description:
      "Extensive inflammatory papules, pustules, and early nodular lesions with elevated erythema.",
    recommendation:
      "Clinical dermatologist evaluation advised for systemic + topical prescription regimen.",
  },
  "Very Severe (Level 3)": {
    level: "Grade 3",
    shortLabel: "Very Severe",
    lesionRange: ">50 inflammatory lesions",
    barGradient: "from-rose-500 to-pink-500",
    ringColor: "#f43f5e",
    badgeBg: "bg-rose-500/15 text-rose-300 border-rose-500/30",
    cardBorder: "border-rose-500/40 bg-rose-950/20",
    dot: "bg-rose-500",
    description:
      "High lesion density with deep nodulocystic involvement and high risk of permanent scarring.",
    recommendation:
      "Prompt specialist dermatology referral recommended for targeted systemic intervention.",
  },
};

export default function App() {
  const [inputMode, setInputMode] = useState("upload"); // "upload" | "webcam"
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [imageDimensions, setImageDimensions] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [heatmapUrl, setHeatmapUrl] = useState(null);

  // Interactive Grad-CAM viewer controls
  const [vizMode, setVizMode] = useState("split"); // "split" | "blend"
  const [blendOpacity, setBlendOpacity] = useState(65);

  // API Health status & session history
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

  const checkBackendHealth = async () => {
    setApiHealth((prev) => ({ ...prev, checking: true }));
    try {
      const res = await fetch(`${API_BASE_URL}/health`);
      if (res.ok) {
        const data = await res.json();
        setApiHealth({
          checking: false,
          online: true,
          modelLoaded: Boolean(data.model_loaded),
        });
      } else {
        setApiHealth({ checking: false, online: false, modelLoaded: false });
      }
    } catch {
      setApiHealth({ checking: false, online: false, modelLoaded: false });
    }
  };

  useEffect(() => {
    checkBackendHealth();
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
      setErrorMessage(
        "Invalid file format. Please upload a JPEG (.jpg) or PNG (.png) image."
      );
      return;
    }

    setErrorMessage(null);
    setPrediction(null);
    setHeatmapUrl(null);
    setImageDimensions(null);

    const objectUrl = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      setImageDimensions({ width: img.naturalWidth, height: img.naturalHeight });
    };
    img.src = objectUrl;

    setSelectedFile(file);
    setPreviewUrl(objectUrl);
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
    setSelectedFile(null);
    setPreviewUrl(null);
    setImageDimensions(null);
    setPrediction(null);
    setHeatmapUrl(null);
    setErrorMessage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const executeInferenceForFile = async (fileToAnalyze, previewForHistory) => {
    if (!fileToAnalyze) {
      setErrorMessage(
        "Please select or capture a dermatological image before running inference."
      );
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
          // Ignore JSON parse errors
        }
        throw new Error(detail);
      }

      const data = await response.json();
      setPrediction(data);
      setApiHealth({ checking: false, online: true, modelLoaded: true });

      let generatedHeatmapUrl = null;
      try {
        const vizFormData = new FormData();
        vizFormData.append("file", fileToAnalyze);
        const vizResponse = await fetch(`${API_BASE_URL}/predict/visualize`, {
          method: "POST",
          body: vizFormData,
        });
        if (vizResponse.ok) {
          const blob = await vizResponse.blob();
          generatedHeatmapUrl = URL.createObjectURL(blob);
          setHeatmapUrl(generatedHeatmapUrl);
        }
      } catch {
        // Keep prediction even if visualization endpoint fails
      }

      setScanHistory((prev) =>
        [
          {
            id: `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
            fileName: fileToAnalyze.name,
            timestamp: new Date().toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
              second: "2-digit",
            }),
            previewUrl: previewForHistory || previewUrl,
            heatmapUrl: generatedHeatmapUrl,
            prediction: data,
          },
          ...prev,
        ].slice(0, 5)
      );
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
    await executeInferenceForFile(selectedFile, previewUrl);
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

    ctx.drawImage(video, 0, 0, width, height);

    canvas.toBlob(
      async (blob) => {
        if (!blob) {
          setErrorMessage("Failed to encode webcam frame as JPEG blob.");
          return;
        }

        const capturedFile = new File([blob], `webcam_${Date.now()}.jpg`, {
          type: "image/jpeg",
        });
        const capturedUrl = URL.createObjectURL(capturedFile);

        setSelectedFile(capturedFile);
        setPreviewUrl(capturedUrl);
        setImageDimensions({ width, height });

        await executeInferenceForFile(capturedFile, capturedUrl);
      },
      "image/jpeg",
      0.95
    );
  };

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

  const activeMeta =
    (prediction && SEVERITY_META[prediction.predicted_class]) ||
    SEVERITY_META["Mild (Level 0)"];

  const topConfidencePct = prediction
    ? (
        (prediction.confidence ??
          prediction.probabilities?.[prediction.predicted_class] ??
          0) * 100
      ).toFixed(1)
    : "0.0";

  // SVG Ring Gauge math
  const ringRadius = 34;
  const ringCircumference = 2 * Math.PI * ringRadius;
  const ringOffset =
    ringCircumference - (Number(topConfidencePct) / 100) * ringCircumference;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col relative overflow-x-hidden">
      {/* Ambient Background Glow */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden z-0">
        <div className="absolute -top-40 -left-40 w-96 h-96 bg-indigo-600/15 rounded-full blur-3xl" />
        <div className="absolute top-20 right-0 w-[28rem] h-[28rem] bg-teal-500/10 rounded-full blur-3xl" />
        <div className="absolute bottom-0 left-1/3 w-96 h-96 bg-violet-600/10 rounded-full blur-3xl" />
      </div>

      {/* Hidden Canvas for Webcam Frame Extraction */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Top Glassmorphic Navigation Bar */}
      <header className="sticky top-0 z-30 border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3.5">
            <div className="relative h-11 w-11 rounded-xl bg-gradient-to-br from-indigo-500 via-indigo-600 to-teal-500 p-[1px] shadow-lg shadow-indigo-500/20">
              <div className="h-full w-full bg-slate-950/90 rounded-[11px] flex items-center justify-center">
                <svg
                  className="w-6 h-6 text-indigo-400"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23-.693L5 14.5m14.8.8l1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0112 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5"
                  />
                </svg>
              </div>
            </div>
            <div>
              <div className="flex items-center gap-2.5">
                <h1 className="text-lg font-bold tracking-tight bg-gradient-to-r from-white via-slate-100 to-slate-400 bg-clip-text text-transparent">
                  SkinWise AI
                </h1>
                <span className="text-[11px] font-mono font-medium uppercase px-2 py-0.5 rounded-full bg-indigo-500/15 text-indigo-300 border border-indigo-500/30">
                  ACNE04 Clinical Suite
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Automated Hayashi Severity Grading &bull; EfficientNet-B0 ONNX &amp; Grad-CAM Explainability
              </p>
            </div>
          </div>

          {/* Right Telemetry Pills */}
          <div className="flex items-center gap-2.5 text-xs">
            <button
              type="button"
              onClick={checkBackendHealth}
              title="Click to refresh backend connection status"
              className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full border font-mono transition cursor-pointer ${
                apiHealth.online
                  ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-300 hover:bg-emerald-500/20"
                  : "bg-rose-500/10 border-rose-500/30 text-rose-300 hover:bg-rose-500/20"
              }`}
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  apiHealth.checking
                    ? "bg-amber-400 animate-ping"
                    : apiHealth.online
                    ? "bg-emerald-400 shadow-[0_0_8px_#34d399]"
                    : "bg-rose-500"
                }`}
              />
              {apiHealth.checking
                ? "Checking API..."
                : apiHealth.online
                ? "ONNX Engine Ready"
                : "API Offline (Retry)"}
            </button>

            <div className="hidden md:inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-slate-900/90 border border-slate-800 text-slate-300 font-mono">
              <span className="text-slate-500">Tensor:</span> 224&times;224 RGB
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Container */}
      <main className="relative z-10 flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        {/* Hayashi 4-Level Severity Reference Strip */}
        <section className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          {CLASS_ORDER.map((clsName) => {
            const meta = SEVERITY_META[clsName];
            const isPredicted = prediction?.predicted_class === clsName;
            return (
              <div
                key={clsName}
                className={`rounded-xl border p-3.5 transition-all ${
                  isPredicted
                    ? `${meta.cardBorder} ring-1 ring-white/20 shadow-lg`
                    : "border-slate-800/80 bg-slate-900/50 hover:bg-slate-900/80"
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-slate-200">
                    <span className={`h-2 w-2 rounded-full ${meta.dot}`} />
                    {meta.shortLabel}
                  </span>
                  <span className="text-[11px] font-mono text-slate-400 px-1.5 py-0.5 rounded bg-slate-800/80">
                    {meta.level}
                  </span>
                </div>
                <p className="text-[11px] font-mono text-slate-400">
                  {meta.lesionRange}
                </p>
              </div>
            );
          })}
        </section>

        {/* Core 12-Column Studio Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left Column: Image Acquisition Studio (5 cols) */}
          <section className="lg:col-span-5 bg-slate-900/70 backdrop-blur-md rounded-2xl border border-slate-800/90 shadow-xl p-6 space-y-5">
            <div className="flex items-center justify-between">
              <div>
                <span className="text-[11px] font-mono uppercase tracking-wider text-indigo-400 font-semibold">
                  Step 01 &bull; Specimen Input
                </span>
                <h2 className="text-base font-bold text-white mt-0.5">
                  Dermatological Image Acquisition
                </h2>
              </div>
              {(selectedFile || prediction) && (
                <button
                  type="button"
                  onClick={handleReset}
                  className="inline-flex items-center gap-1 text-xs font-medium text-slate-400 hover:text-rose-400 transition cursor-pointer px-2.5 py-1 rounded-lg hover:bg-rose-500/10 border border-transparent hover:border-rose-500/20"
                >
                  <svg
                    className="w-3.5 h-3.5"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={2}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                  Reset
                </button>
              )}
            </div>

            {/* Segmented Mode Switcher */}
            <div className="grid grid-cols-2 gap-1.5 p-1.5 bg-slate-950/90 rounded-xl border border-slate-800">
              <button
                type="button"
                onClick={() => handleSwitchMode("upload")}
                className={`py-2.5 px-3 rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition-all cursor-pointer ${
                  inputMode === "upload"
                    ? "bg-gradient-to-r from-indigo-600 to-indigo-500 text-white shadow-md shadow-indigo-500/20"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <svg
                  className="w-4 h-4"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
                  />
                </svg>
                Upload Image
              </button>

              <button
                type="button"
                onClick={() => handleSwitchMode("webcam")}
                className={`py-2.5 px-3 rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition-all cursor-pointer ${
                  inputMode === "webcam"
                    ? "bg-gradient-to-r from-indigo-600 to-indigo-500 text-white shadow-md shadow-indigo-500/20"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <span
                  className={`h-2 w-2 rounded-full ${
                    isCameraActive ? "bg-rose-400 animate-ping" : "bg-slate-500"
                  }`}
                />
                Live Camera
              </button>
            </div>

            {inputMode === "upload" ? (
              <>
                {/* Interactive Drag & Drop Zone */}
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
                  className={`relative group cursor-pointer rounded-2xl border-2 border-dashed transition-all duration-200 p-5 text-center overflow-hidden ${
                    isDragging
                      ? "border-indigo-400 bg-indigo-500/10 scale-[1.01]"
                      : selectedFile
                      ? "border-slate-700 bg-slate-950/60 hover:border-indigo-500/60"
                      : "border-slate-800 bg-slate-950/50 hover:bg-slate-950/90 hover:border-indigo-500/50"
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
                    <div className="space-y-4">
                      <div className="relative mx-auto w-full max-w-[280px] aspect-square rounded-xl overflow-hidden border border-slate-700/80 bg-slate-950 shadow-2xl group">
                        <img
                          src={previewUrl}
                          alt="Uploaded skin specimen preview"
                          className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-slate-950/90 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-end justify-center p-3">
                          <span className="text-xs font-medium text-white bg-slate-900/90 px-3 py-1.5 rounded-full border border-slate-700">
                            Click or drop to replace image
                          </span>
                        </div>
                      </div>

                      {/* File Metadata Pills */}
                      <div className="flex flex-wrap items-center justify-center gap-2 text-xs font-mono">
                        <span className="px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-slate-300 truncate max-w-[200px]">
                          {selectedFile?.name}
                        </span>
                        <span className="px-2 py-1 rounded-md bg-slate-900 border border-slate-800 text-slate-400">
                          {((selectedFile?.size || 0) / 1024).toFixed(1)} KB
                        </span>
                        {imageDimensions && (
                          <span className="px-2 py-1 rounded-md bg-indigo-500/10 border border-indigo-500/30 text-indigo-300">
                            {imageDimensions.width}&times;{imageDimensions.height}px
                          </span>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div className="py-10 px-4 space-y-4">
                      <div className="mx-auto h-16 w-16 rounded-2xl bg-gradient-to-br from-indigo-500/20 to-teal-500/10 border border-indigo-500/30 text-indigo-400 flex items-center justify-center shadow-inner group-hover:scale-110 transition-transform">
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          className="h-8 w-8"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                          strokeWidth={1.75}
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                          />
                        </svg>
                      </div>
                      <div className="space-y-1.5">
                        <p className="text-sm font-semibold text-slate-200">
                          Drop a facial skin image here, or{" "}
                          <span className="text-indigo-400 underline decoration-indigo-400/50 underline-offset-4">
                            browse files
                          </span>
                        </p>
                        <p className="text-xs text-slate-400">
                          Supports high-res JPEG &amp; PNG &bull; Auto-normalized to 224&times;224
                        </p>
                      </div>
                      <div className="pt-2 flex items-center justify-center gap-2 text-[11px] font-mono text-slate-500">
                        <span className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800">
                          acne0_1024
                        </span>
                        <span className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800">
                          acne1_1024
                        </span>
                        <span className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800">
                          acne2_1024
                        </span>
                        <span className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800">
                          acne3_1024
                        </span>
                      </div>
                    </div>
                  )}
                </div>

                {/* Primary Run Inference Button */}
                <button
                  type="button"
                  onClick={handleRunInference}
                  disabled={!selectedFile || isLoading}
                  className={`w-full py-3.5 px-5 rounded-xl font-semibold text-sm flex items-center justify-center gap-2.5 transition-all ${
                    !selectedFile || isLoading
                      ? "bg-slate-800/80 text-slate-500 border border-slate-800 cursor-not-allowed"
                      : "bg-gradient-to-r from-indigo-600 via-indigo-500 to-teal-500 text-white hover:brightness-110 active:scale-[0.99] shadow-lg shadow-indigo-600/25 cursor-pointer"
                  }`}
                >
                  {isLoading ? (
                    <>
                      <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                      <span>Running ONNX Inference &amp; Grad-CAM...</span>
                    </>
                  ) : (
                    <>
                      <svg
                        className="w-4 h-4"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                        strokeWidth={2}
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M13 10V3L4 14h7v7l9-11h-7z"
                        />
                      </svg>
                      <span>Analyze Skin Severity</span>
                    </>
                  )}
                </button>
              </>
            ) : (
              /* Live Webcam Capture Viewfinder */
              <div className="space-y-4">
                <div className="relative rounded-2xl overflow-hidden border border-slate-700/80 bg-slate-950 aspect-square flex items-center justify-center shadow-inner">
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className={`w-full h-full object-cover ${
                      isCameraActive ? "block" : "hidden"
                    }`}
                  />

                  {/* Viewfinder Framing Brackets when active */}
                  {isCameraActive && (
                    <div className="pointer-events-none absolute inset-8 border border-white/15 rounded-2xl">
                      <div className="absolute -top-0.5 -left-0.5 w-5 h-5 border-t-2 border-l-2 border-teal-400 rounded-tl" />
                      <div className="absolute -top-0.5 -right-0.5 w-5 h-5 border-t-2 border-r-2 border-teal-400 rounded-tr" />
                      <div className="absolute -bottom-0.5 -left-0.5 w-5 h-5 border-b-2 border-l-2 border-teal-400 rounded-bl" />
                      <div className="absolute -bottom-0.5 -right-0.5 w-5 h-5 border-b-2 border-r-2 border-teal-400 rounded-br" />
                    </div>
                  )}

                  {!isCameraActive && (
                    <div className="text-center p-6 space-y-3">
                      <div className="mx-auto h-12 w-12 rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center text-slate-400">
                        <svg
                          className="w-6 h-6"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                          strokeWidth={1.75}
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                          />
                        </svg>
                      </div>
                      <p className="text-xs text-slate-400">
                        Camera stream is currently paused.
                      </p>
                      <button
                        type="button"
                        onClick={startCameraStream}
                        className="px-4 py-2 rounded-xl bg-indigo-600 text-white text-xs font-semibold hover:bg-indigo-500 cursor-pointer shadow"
                      >
                        Enable Camera
                      </button>
                    </div>
                  )}

                  {isCameraActive && (
                    <div className="absolute top-3 left-3 inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-950/80 border border-slate-700 text-white text-[11px] font-mono backdrop-blur-md">
                      <span className="h-2 w-2 rounded-full bg-rose-500 animate-ping" />
                      LIVE OPTICAL FEED
                    </div>
                  )}
                </div>

                <div className="flex gap-2.5">
                  <button
                    type="button"
                    onClick={handleCaptureAndAnalyze}
                    disabled={!isCameraActive || isLoading}
                    className={`flex-1 py-3.5 px-4 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all ${
                      !isCameraActive || isLoading
                        ? "bg-slate-800 text-slate-500 cursor-not-allowed"
                        : "bg-gradient-to-r from-indigo-600 to-teal-500 text-white hover:brightness-110 shadow-lg shadow-indigo-600/20 cursor-pointer"
                    }`}
                  >
                    {isLoading ? "Analyzing Frame..." : "Capture & Analyze Frame"}
                  </button>

                  {isCameraActive ? (
                    <button
                      type="button"
                      onClick={stopCameraStream}
                      className="px-4 py-3.5 rounded-xl border border-slate-700 bg-slate-900 text-slate-300 hover:bg-slate-800 text-xs font-semibold cursor-pointer"
                    >
                      Stop
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={startCameraStream}
                      className="px-4 py-3.5 rounded-xl border border-slate-700 bg-slate-900 text-slate-300 hover:bg-slate-800 text-xs font-semibold cursor-pointer"
                    >
                      Start
                    </button>
                  )}
                </div>
              </div>
            )}

            {/* Error Alert Banner */}
            {errorMessage && (
              <div
                role="alert"
                className="rounded-xl bg-rose-500/10 border border-rose-500/30 px-4 py-3 text-xs text-rose-200 flex items-start gap-2.5"
              >
                <svg
                  className="w-4 h-4 text-rose-400 shrink-0 mt-0.5"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                  />
                </svg>
                <div>
                  <span className="font-bold uppercase tracking-wider text-rose-400 mr-1.5">
                    Diagnostic Error:
                  </span>
                  <span>{errorMessage}</span>
                </div>
              </div>
            )}

            {/* Recent Scans Session Strip */}
            {scanHistory.length > 0 && (
              <div className="pt-4 border-t border-slate-800/80 space-y-2.5">
                <div className="flex items-center justify-between text-[11px] font-mono uppercase tracking-wider text-slate-400">
                  <span>Recent Session Scans ({scanHistory.length})</span>
                  <span>Click to recall</span>
                </div>
                <div className="flex gap-2.5 overflow-x-auto pb-1">
                  {scanHistory.map((item) => {
                    const itemMeta =
                      SEVERITY_META[item.prediction.predicted_class] ||
                      SEVERITY_META["Mild (Level 0)"];
                    return (
                      <button
                        key={item.id}
                        type="button"
                        onClick={() => {
                          setPreviewUrl(item.previewUrl);
                          setHeatmapUrl(item.heatmapUrl);
                          setPrediction(item.prediction);
                        }}
                        className="group shrink-0 flex items-center gap-2.5 p-2 rounded-xl bg-slate-950/80 border border-slate-800 hover:border-indigo-500/50 transition text-left cursor-pointer"
                      >
                        {item.previewUrl && (
                          <img
                            src={item.previewUrl}
                            alt={item.fileName}
                            className="w-10 h-10 rounded-lg object-cover border border-slate-700"
                          />
                        )}
                        <div className="pr-1">
                          <div className="flex items-center gap-1.5">
                            <span
                              className={`h-2 w-2 rounded-full ${itemMeta.dot}`}
                            />
                            <span className="text-xs font-semibold text-slate-200">
                              {itemMeta.shortLabel}
                            </span>
                          </div>
                          <span className="block text-[10px] font-mono text-slate-500">
                            {item.timestamp}
                          </span>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
          </section>

          {/* Right Column: Diagnostic Report & Grad-CAM Studio (7 cols) */}
          <section className="lg:col-span-7 bg-slate-900/70 backdrop-blur-md rounded-2xl border border-slate-800/90 shadow-xl p-6 space-y-6">
            <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-slate-800/80">
              <div>
                <span className="text-[11px] font-mono uppercase tracking-wider text-teal-400 font-semibold">
                  Step 02 &bull; Quantitative Assessment
                </span>
                <h2 className="text-base font-bold text-white mt-0.5">
                  Diagnostic Grading &amp; Grad-CAM Localization
                </h2>
              </div>

              {/* Execution Latency Telemetry Pill */}
              {prediction && typeof prediction.latency_ms === "number" && (
                <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-slate-950 border border-slate-800 text-xs font-mono">
                  <span className="h-2 w-2 rounded-full bg-teal-400" />
                  <span className="text-slate-400">ONNX Runtime:</span>
                  <span className="font-semibold text-teal-300">
                    {prediction.latency_ms.toFixed(2)} ms
                  </span>
                </div>
              )}
            </div>

            {/* Empty State: Visual 3-Stage Pipeline Guide */}
            {!prediction && !isLoading && (
              <div className="py-10 px-4 space-y-8">
                <div className="text-center max-w-md mx-auto space-y-2">
                  <div className="inline-flex items-center justify-center w-12 h-12 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 mb-1">
                    <svg
                      className="w-6 h-6"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={1.75}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                      />
                    </svg>
                  </div>
                  <h3 className="text-base font-bold text-slate-200">
                    Ready for Dermatological Specimen Analysis
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    Select a skin image from your dataset or capture a frame from the live camera to compute 4-class Hayashi probabilities and spatial Grad-CAM heatmaps.
                  </p>
                </div>

                {/* 3 Pipeline Step Cards */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3.5">
                  <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800/90 space-y-1.5">
                    <span className="text-[11px] font-mono text-indigo-400 font-semibold">
                      01 / PREPROCESS
                    </span>
                    <h4 className="text-xs font-bold text-slate-200">
                      ImageNet Normalization
                    </h4>
                    <p className="text-[11px] text-slate-400 leading-relaxed">
                      Bilinear resize to 224&times;224 RGB tensor with standard channel mean/std scaling.
                    </p>
                  </div>

                  <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800/90 space-y-1.5">
                    <span className="text-[11px] font-mono text-teal-400 font-semibold">
                      02 / INFERENCE
                    </span>
                    <h4 className="text-xs font-bold text-slate-200">
                      EfficientNet-B0 ONNX
                    </h4>
                    <p className="text-[11px] text-slate-400 leading-relaxed">
                      Sub-20ms CPU execution computing posterior softmax probabilities across 4 grades.
                    </p>
                  </div>

                  <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800/90 space-y-1.5">
                    <span className="text-[11px] font-mono text-violet-400 font-semibold">
                      03 / EXPLAINABILITY
                    </span>
                    <h4 className="text-xs font-bold text-slate-200">
                      Mathematical Grad-CAM
                    </h4>
                    <p className="text-[11px] text-slate-400 leading-relaxed">
                      Gradients pooled from the final convolutional layer highlight inflammatory zones.
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Loading State */}
            {isLoading && (
              <div className="py-20 text-center space-y-4">
                <div className="relative mx-auto w-14 h-14 flex items-center justify-center">
                  <div className="absolute inset-0 rounded-full border-2 border-indigo-500/20" />
                  <div className="absolute inset-0 rounded-full border-2 border-indigo-400 border-t-transparent animate-spin" />
                  <span className="text-xs font-mono text-indigo-300">AI</span>
                </div>
                <div className="space-y-1">
                  <p className="text-sm font-semibold text-slate-200">
                    Executing EfficientNet-B0 ONNX Graph &amp; Grad-CAM Hooks...
                  </p>
                  <p className="text-xs text-slate-500 font-mono">
                    POST {API_BASE_URL}/predict &bull; /predict/visualize
                  </p>
                </div>
              </div>
            )}

            {/* Active Prediction Output */}
            {prediction && !isLoading && (
              <div className="space-y-6">
                {/* Top Hero Banner with SVG Confidence Gauge */}
                <div
                  className={`rounded-2xl border p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-5 ${activeMeta.cardBorder}`}
                >
                  <div className="space-y-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <span
                        className={`inline-flex items-center gap-1.5 text-xs font-mono font-semibold uppercase px-2.5 py-0.5 rounded-full border ${activeMeta.badgeBg}`}
                      >
                        <span className={`h-2 w-2 rounded-full ${activeMeta.dot}`} />
                        {activeMeta.level} &bull; {activeMeta.lesionRange}
                      </span>
                      <span className="text-[11px] font-mono text-slate-400">
                        Class Index #{prediction.predicted_class_index}
                      </span>
                    </div>

                    <h3 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-white">
                      {prediction.predicted_class}
                    </h3>

                    <p className="text-xs text-slate-300 leading-relaxed max-w-xl">
                      {activeMeta.description}
                    </p>

                    <div className="pt-1 flex items-center gap-2 text-xs text-slate-300/90">
                      <span className="font-mono uppercase text-[10px] px-2 py-0.5 rounded bg-white/10 text-white font-semibold">
                        Clinical Note
                      </span>
                      <span>{activeMeta.recommendation}</span>
                    </div>
                  </div>

                  {/* Circular Confidence Ring */}
                  <div className="shrink-0 flex items-center gap-3 sm:flex-col sm:justify-center bg-slate-950/75 border border-white/10 rounded-2xl px-4 py-3">
                    <div className="relative w-20 h-20 flex items-center justify-center">
                      <svg className="w-20 h-20 -rotate-90" viewBox="0 0 80 80">
                        <circle
                          cx="40"
                          cy="40"
                          r={ringRadius}
                          stroke="currentColor"
                          strokeWidth="7"
                          fill="transparent"
                          className="text-slate-800"
                        />
                        <circle
                          cx="40"
                          cy="40"
                          r={ringRadius}
                          stroke={activeMeta.ringColor}
                          strokeWidth="7"
                          strokeDasharray={ringCircumference}
                          strokeDashoffset={ringOffset}
                          strokeLinecap="round"
                          fill="transparent"
                          className="transition-all duration-700 ease-out"
                        />
                      </svg>
                      <div className="absolute inset-0 flex flex-col items-center justify-center">
                        <span className="text-sm font-bold font-mono text-white">
                          {topConfidencePct}%
                        </span>
                      </div>
                    </div>
                    <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400">
                      Top-1 Confidence
                    </span>
                  </div>
                </div>

                {/* Interactive Grad-CAM Visual Explainability Studio */}
                {(previewUrl || heatmapUrl) && (
                  <div className="rounded-2xl bg-slate-950/70 border border-slate-800/90 p-5 space-y-4">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div>
                        <h4 className="text-xs font-mono uppercase tracking-wider text-indigo-400 font-semibold">
                          Grad-CAM Spatial Activation Map
                        </h4>
                        <p className="text-xs text-slate-400 mt-0.5">
                          Regions with highest convolutional gradient contribution to{" "}
                          <span className="text-slate-200 font-medium">
                            {prediction.predicted_class}
                          </span>
                        </p>
                      </div>

                      {/* View Mode Toggle: Split vs Interactive Blend */}
                      {previewUrl && heatmapUrl && (
                        <div className="inline-flex p-1 rounded-xl bg-slate-900 border border-slate-800 text-xs">
                          <button
                            type="button"
                            onClick={() => setVizMode("split")}
                            className={`px-3 py-1 rounded-lg font-medium transition cursor-pointer ${
                              vizMode === "split"
                                ? "bg-indigo-600 text-white shadow"
                                : "text-slate-400 hover:text-slate-200"
                            }`}
                          >
                            Side-by-Side
                          </button>
                          <button
                            type="button"
                            onClick={() => setVizMode("blend")}
                            className={`px-3 py-1 rounded-lg font-medium transition cursor-pointer ${
                              vizMode === "blend"
                                ? "bg-indigo-600 text-white shadow"
                                : "text-slate-400 hover:text-slate-200"
                            }`}
                          >
                            Interactive Blend
                          </button>
                        </div>
                      )}
                    </div>

                    {vizMode === "split" || !heatmapUrl ? (
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                        {previewUrl && (
                          <div className="space-y-2">
                            <div className="relative aspect-square w-full rounded-xl overflow-hidden border border-slate-800 bg-slate-900 shadow-lg">
                              <img
                                src={previewUrl}
                                alt="Original RGB input"
                                className="w-full h-full object-cover"
                              />
                              <span className="absolute top-2.5 left-2.5 px-2.5 py-1 rounded-md bg-slate-950/80 backdrop-blur-md border border-slate-700 text-[11px] font-mono text-slate-200">
                                Original RGB Specimen
                              </span>
                            </div>
                          </div>
                        )}

                        {heatmapUrl && (
                          <div className="space-y-2">
                            <div className="relative aspect-square w-full rounded-xl overflow-hidden border border-indigo-500/30 bg-slate-900 shadow-lg">
                              <img
                                src={heatmapUrl}
                                alt="Grad-CAM heatmap overlay"
                                className="w-full h-full object-cover"
                              />
                              <span className="absolute top-2.5 left-2.5 px-2.5 py-1 rounded-md bg-slate-950/80 backdrop-blur-md border border-indigo-500/40 text-[11px] font-mono text-indigo-300">
                                Grad-CAM Overlay (224&times;224)
                              </span>
                            </div>
                          </div>
                        )}
                      </div>
                    ) : (
                      /* Interactive Opacity Blend Slider View */
                      <div className="space-y-4">
                        <div className="relative mx-auto max-w-sm aspect-square w-full rounded-xl overflow-hidden border border-indigo-500/40 bg-slate-900 shadow-2xl">
                          <img
                            src={previewUrl}
                            alt="Original RGB base"
                            className="w-full h-full object-cover"
                          />
                          <img
                            src={heatmapUrl}
                            alt="Grad-CAM overlay layer"
                            style={{ opacity: blendOpacity / 100 }}
                            className="absolute inset-0 w-full h-full object-cover transition-opacity duration-75"
                          />
                          <span className="absolute top-3 left-3 px-2.5 py-1 rounded-md bg-slate-950/85 border border-slate-700 text-[11px] font-mono text-slate-200">
                            Heatmap Opacity: {blendOpacity}%
                          </span>
                        </div>

                        <div className="max-w-sm mx-auto flex items-center gap-3 text-xs font-mono text-slate-400">
                          <span>RGB (0%)</span>
                          <input
                            type="range"
                            min={0}
                            max={100}
                            value={blendOpacity}
                            onChange={(e) => setBlendOpacity(Number(e.target.value))}
                            className="flex-1 accent-indigo-500 cursor-pointer"
                          />
                          <span>Grad-CAM (100%)</span>
                        </div>
                      </div>
                    )}

                    {/* Jet Colormap Scale Bar */}
                    <div className="pt-2 flex items-center justify-between gap-3 text-[11px] font-mono text-slate-400">
                      <span>Low Activation (0.0)</span>
                      <div className="flex-1 max-w-xs h-2 rounded-full bg-gradient-to-r from-blue-600 via-emerald-400 via-yellow-400 to-rose-600 border border-white/10" />
                      <span>High Lesion Salience (1.0)</span>
                    </div>
                  </div>
                )}

                {/* 4-Class Softmax Probability Distribution */}
                <div className="space-y-3.5">
                  <div className="flex items-center justify-between text-xs font-mono uppercase tracking-wider text-slate-400">
                    <span>Posterior Softmax Distribution (4 Classes)</span>
                    <span>Probability (%)</span>
                  </div>

                  <div className="grid grid-cols-1 gap-3">
                    {probabilityEntries.map(([className, probValue]) => {
                      const pct = Math.max(
                        0,
                        Math.min(100, Number(probValue) * 100)
                      );
                      const isWinner = className === prediction.predicted_class;
                      const meta =
                        SEVERITY_META[className] ||
                        SEVERITY_META["Mild (Level 0)"];

                      return (
                        <div
                          key={className}
                          className={`p-3.5 rounded-xl border transition-all ${
                            isWinner
                              ? "border-indigo-500/50 bg-slate-950 shadow-md"
                              : "border-slate-800/80 bg-slate-950/40"
                          }`}
                        >
                          <div className="flex items-center justify-between text-sm mb-2">
                            <div className="flex items-center gap-2.5">
                              <span
                                className={`h-2.5 w-2.5 rounded-full ${meta.dot}`}
                              />
                              <span
                                className={
                                  isWinner
                                    ? "font-bold text-white"
                                    : "font-medium text-slate-300"
                                }
                              >
                                {className}
                              </span>
                              <span className="text-[11px] font-mono text-slate-500 hidden sm:inline">
                                ({meta.lesionRange})
                              </span>
                              {isWinner && (
                                <span className="text-[10px] font-mono font-semibold uppercase px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                                  Argmax Winner
                                </span>
                              )}
                            </div>
                            <span className="font-mono text-xs font-bold text-slate-200">
                              {pct.toFixed(2)}%
                            </span>
                          </div>

                          <div className="w-full h-2.5 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
                            <div
                              className={`h-full rounded-full bg-gradient-to-r ${meta.barGradient} transition-all duration-500 ease-out`}
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
              </div>
            )}
          </section>
        </div>
      </main>

      {/* Footer */}
      <footer className="relative z-10 border-t border-slate-900 bg-slate-950/90 py-4 px-6 text-center text-xs text-slate-500">
        SkinWise AI &mdash; Deep Learning &amp; MLOps Clinical Grading Suite (ACNE04 Benchmark). Research prototype; not a substitute for professional dermatological advice.
      </footer>
    </div>
  );
}
