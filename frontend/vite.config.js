import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { compile, optimize } from "@tailwindcss/node";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function extractCandidatesFromFiles(rootDir) {
  const files = [
    path.join(rootDir, "index.html"),
    path.join(rootDir, "src", "App.jsx"),
    path.join(rootDir, "src", "main.jsx"),
  ];
  const candidates = new Set();
  const tokenRegex = /[^\s"'`<>={}();,]+/g;

  for (const file of files) {
    if (!fs.existsSync(file)) continue;
    const content = fs.readFileSync(file, "utf8");
    const matches = content.match(tokenRegex) || [];
    for (const token of matches) {
      const cleaned = token.replace(/^[:('${}`]+|[:)'"${}`]+$/g, "");
      if (cleaned) candidates.add(cleaned);
    }
  }
  return [...candidates];
}

function reliableTailwindV4Plugin() {
  return {
    name: "skinwise-tailwind-v4",
    enforce: "pre",
    async transform(code, id) {
      if (!id.endsWith(".css")) return null;
      const baseDir = path.dirname(id.split("?")[0]);
      const compiler = await compile(code, {
        base: baseDir,
        onDependency: () => {},
      });
      const candidates = extractCandidatesFromFiles(__dirname);
      const builtCss = compiler.build(candidates);
      const optimized = optimize(builtCss, { minify: false });
      return { code: optimized.code, map: null };
    },
  };
}

export default defineConfig({
  plugins: [react(), reliableTailwindV4Plugin()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: process.env.BACKEND_INTERNAL_URL || "http://backend:8000",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
});
