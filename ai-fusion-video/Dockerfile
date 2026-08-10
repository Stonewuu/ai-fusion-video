# ============================================
# 融光 - 后端 Dockerfile (多阶段构建)
# ============================================

# ---------- 阶段 1: 构建 ----------
FROM eclipse-temurin:21-jdk AS builder

WORKDIR /build

# 先复制 Maven Wrapper 和 pom.xml，利用 Docker 缓存加速依赖下载
COPY .mvn/ .mvn/
COPY mvnw pom.xml ./
# 兼容 Windows CRLF 行尾（即使不经过 Git checkout）
RUN sed -i 's/\r$//' mvnw && chmod +x mvnw
RUN --mount=type=cache,target=/root/.m2 ./mvnw dependency:go-offline -B

# 复制源码并打包
COPY src/ src/
RUN --mount=type=cache,target=/root/.m2 ./mvnw package -DskipTests -B

# ---------- 阶段 2: 运行 ----------
FROM eclipse-temurin:21-jre

LABEL maintainer="Stonewu <https://github.com/Stonewuu>"

WORKDIR /app

RUN set -eux; \
  if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \
    sed -i 's|http://archive.ubuntu.com/ubuntu/|https://archive.ubuntu.com/ubuntu/|g; s|http://security.ubuntu.com/ubuntu/|https://security.ubuntu.com/ubuntu/|g' /etc/apt/sources.list.d/ubuntu.sources; \
  fi; \
  if [ -f /etc/apt/sources.list ]; then \
    sed -i 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' /etc/apt/sources.list; \
  fi; \
  apt-get update -o Acquire::Retries=5; \
  apt-get install -y --no-install-recommends ffmpeg curl; \
  rm -rf /var/lib/apt/lists/*

# 从构建阶段复制 JAR
COPY --from=builder /build/target/*.jar app.jar

# 创建数据目录
RUN mkdir -p /app/data/media /app/data/agent-workspace

# 暴露端口
EXPOSE 18080

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:18080/v3/api-docs || exit 1

ENTRYPOINT ["java", "-jar", "app.jar"]
