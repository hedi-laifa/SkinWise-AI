/**
 * SkinWise AI — API service layer
 * Centralises all fetch calls to the FastAPI backend.
 */

const API_BASE_URL =
  (typeof import.meta !== "undefined" &&
    import.meta.env &&
    import.meta.env.VITE_API_BASE_URL) ||
  "http://localhost:8000";

/**
 * Check backend health and model status.
 * @returns {{ online: boolean, modelLoaded: boolean, gpuAvailable: boolean }}
 */
export async function checkHealth() {
  const res = await fetch(`${API_BASE_URL}/health`);
  if (!res.ok) throw new Error(`Health check failed: HTTP ${res.status}`);
  const data = await res.json();
  return {
    online: true,
    modelLoaded: Boolean(data.model_loaded),
    gpuAvailable: Boolean(data.gpu_available),
    gpuDevice: data.gpu_device ?? null,
  };
}

/**
 * Run acne severity inference on an image file.
 * @param {File} file  JPEG or PNG image
 * @returns {Promise<Object>} PredictionResponse JSON
 */
export async function predict(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    let detail = `Server responded with HTTP ${res.status}`;
    try {
      const err = await res.json();
      if (err?.detail) detail = err.detail;
    } catch {
      // Ignore JSON parse errors
    }
    throw new Error(detail);
  }

  return res.json();
}

/**
 * Request a Grad-CAM heatmap overlay for an image file.
 * @param {File} file  JPEG or PNG image
 * @returns {Promise<string>} Object URL for the returned PNG blob
 */
export async function visualize(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE_URL}/predict/visualize`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) throw new Error(`Visualize failed: HTTP ${res.status}`);
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

/**
 * Fetch model architecture metadata.
 * @returns {Promise<Object>} ModelMetadataResponse JSON
 */
export async function getModelMetadata() {
  const res = await fetch(`${API_BASE_URL}/model/metadata`);
  if (!res.ok) throw new Error(`Metadata fetch failed: HTTP ${res.status}`);
  return res.json();
}

export { API_BASE_URL };
