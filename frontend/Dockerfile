# ==============================================================================
# SkinWise AI - Production frontend image (React build served by nginx)
# ==============================================================================

# ------------------------- Stage 1: Dependencies ------------------------------
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

# ------------------------- Stage 2: Build -------------------------------------
# Vite compiles the React app into static files (HTML, JS, CSS) in /app/dist.
FROM deps AS builder
WORKDIR /app
COPY . .
# The API address is embedded in the JavaScript at build time
ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build

# ------------------------- Stage 3: Runtime (nginx) ---------------------------
# Only the static files are kept: no Node.js, no node_modules, no source code.
FROM nginx:1.27-alpine AS runtime
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD ["wget", "-q", "--spider", "http://127.0.0.1/"]
