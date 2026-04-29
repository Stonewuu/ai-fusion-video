# ============================================
# 融光 - 前端 Dockerfile (多阶段构建)
# ============================================

# ---------- 阶段 1: 安装依赖 ----------
FROM node:20-alpine AS deps

WORKDIR /build

COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
RUN corepack enable pnpm
RUN --mount=type=cache,target=/root/.local/share/pnpm/store pnpm install --frozen-lockfile

# ---------- 阶段 2: 构建 ----------
FROM node:20-alpine AS builder

WORKDIR /build

COPY --from=deps /build/node_modules ./node_modules
COPY . .

# 构建时注入后端 API 地址（NEXT_PUBLIC_ 变量在 build 时烘焙进客户端 JS）
ARG NEXT_PUBLIC_API_BASE_URL=/api
ENV NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL}
ENV NEXT_TELEMETRY_DISABLED=1

RUN corepack enable pnpm
RUN --mount=type=cache,target=/build/.next/cache pnpm build

# ---------- 阶段 3: 运行 ----------
FROM node:20-alpine

LABEL maintainer="Stonewu <https://github.com/Stonewuu>"

WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

# 复制构建产物
COPY --from=builder /build/.next/standalone ./
COPY --from=builder /build/.next/static ./.next/static
COPY --from=builder /build/public ./public

EXPOSE 3000

CMD ["node", "server.js"]
