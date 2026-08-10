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

COPY app.jar app.jar

RUN mkdir -p /app/data/media /app/data/agent-workspace

EXPOSE 18080

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:18080/v3/api-docs || exit 1

ENTRYPOINT ["java", "-jar", "app.jar"]
