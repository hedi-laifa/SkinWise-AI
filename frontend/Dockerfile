# ==============================================================================
# SkinWise AI - Multi-Stage React / Vite Frontend Dockerfile
# ==============================================================================

# ------------------------- Stage 1: Dependencies ------------------------------
FROM node:20-alpine AS deps

WORKDIR /app
COPY package*.json ./
RUN npm install

# ------------------------- Stage 2: Builder (Verify Build) --------------------
FROM deps AS builder

WORKDIR /app
COPY . .
RUN npm run build

# ------------------------- Stage 3: Runtime (Vite Server on 5173) -------------
FROM node:20-alpine AS runtime

WORKDIR /app

# Copy cached node_modules from deps stage and source files
COPY --from=deps /app/node_modules ./node_modules
COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
