FROM eclipse-temurin:21-jre

LABEL maintainer="Stonewu <https://github.com/Stonewuu>"

WORKDIR /app

COPY app.jar app.jar

RUN mkdir -p /app/data/media

EXPOSE 18080

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:18080/v3/api-docs || exit 1

ENTRYPOINT ["java", "-jar", "app.jar"]