package com.stonewu.fusion.service.ai.googleflow;

import cn.hutool.core.util.StrUtil;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.storage.StorageConfig;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import com.stonewu.fusion.service.storage.StorageConfigService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import okhttp3.ResponseBody;
import okio.BufferedSource;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * Google Flow Reverse API OpenAI 兼容请求客户端。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class GoogleFlowReverseApiClient {

    private static final MediaType JSON = MediaType.get("application/json; charset=utf-8");
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
    private static final String DEFAULT_LOCAL_MEDIA_BASE_PATH = "./data/media";

    private final StorageConfigService storageConfigService;

    private final OkHttpClient streamHttpClient = new OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(0, TimeUnit.MILLISECONDS)
            .writeTimeout(60, TimeUnit.SECONDS)
            .build();

    private final OkHttpClient downloadHttpClient = new OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(120, TimeUnit.SECONDS)
            .writeTimeout(60, TimeUnit.SECONDS)
            .followRedirects(true)
            .build();

    public CompletionResult generate(String prompt, String modelCode, List<String> imageUrls, ApiConfig apiConfig) {
        String baseUrl = GoogleFlowReverseApiSupport.requireBaseUrl(apiConfig);
        if (StrUtil.isBlank(apiConfig.getApiKey())) {
            throw new BusinessException("GoogleFlowReverseApi 缺少 API Key 配置");
        }

        try {
            String requestBody = buildRequestBody(prompt, modelCode, imageUrls);
            return executeStreamingRequest(baseUrl, apiConfig.getApiKey(), requestBody, apiConfig);
        } catch (IOException e) {
            log.error("[GoogleFlowReverseApi] 请求失败: model={}", modelCode, e);
            throw new BusinessException("GoogleFlowReverseApi 请求失败: " + e.getMessage());
        }
    }

    private String buildRequestBody(String prompt, String modelCode, List<String> imageUrls) throws IOException {
        Map<String, Object> userMessage;
        if (imageUrls == null || imageUrls.isEmpty()) {
            userMessage = Map.of(
                    "role", "user",
                    "content", StrUtil.blankToDefault(prompt, "请根据提示生成内容")
            );
        } else {
            List<Map<String, Object>> content = new ArrayList<>();
            content.add(Map.of(
                    "type", "text",
                    "text", StrUtil.blankToDefault(prompt, "请根据提示生成内容")
            ));
            for (String imageUrl : imageUrls) {
                content.add(Map.of(
                        "type", "image_url",
                        "image_url", Map.of("url", toDataUrl(imageUrl))
                ));
            }
            userMessage = Map.of(
                    "role", "user",
                    "content", content
            );
        }

        Map<String, Object> body = Map.of(
                "model", modelCode,
                "messages", List.of(userMessage),
                "stream", true
        );
        return OBJECT_MAPPER.writeValueAsString(body);
    }

    private CompletionResult executeStreamingRequest(String baseUrl, String apiKey, String requestBody,
                                                     ApiConfig apiConfig) throws IOException {
        Request request = new Request.Builder()
                .url(baseUrl + "/v1/chat/completions")
                .post(RequestBody.create(requestBody, JSON))
                .addHeader("Authorization", "Bearer " + apiKey)
                .addHeader("Content-Type", "application/json")
                .addHeader("Accept", "text/event-stream")
                .build();

        OkHttpClient client = AiProxySupport.okHttpClient(streamHttpClient, apiConfig);
        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                String errorBody = response.body() != null ? response.body().string() : "";
                throw new BusinessException("GoogleFlowReverseApi 请求失败: HTTP " + response.code() + " " + extractErrorMessage(errorBody));
            }
            if (response.body() == null) {
                throw new BusinessException("GoogleFlowReverseApi 响应体为空");
            }
            return parseStreamingBody(response.body());
        }
    }

    private CompletionResult parseStreamingBody(ResponseBody body) throws IOException {
        BufferedSource source = body.source();
        String requestId = null;
        String finalContent = null;
        List<String> eventLines = new ArrayList<>();

        while (!source.exhausted()) {
            String line = source.readUtf8Line();
            if (line == null) {
                break;
            }

            if (line.isBlank()) {
                String eventPayload = flushEvent(eventLines);
                if (eventPayload == null) {
                    continue;
                }
                if ("[DONE]".equals(eventPayload)) {
                    break;
                }

                JsonNode payload = OBJECT_MAPPER.readTree(eventPayload);
                if (payload.has("error")) {
                    JsonNode error = payload.get("error");
                    throw new BusinessException("GoogleFlowReverseApi 生成失败: " + textOf(error.get("message"), textOf(error.get("detail"), "未知错误")));
                }

                requestId = textOf(payload.get("id"), requestId);
                String directUrl = textOf(payload.get("url"), null);

                JsonNode choices = payload.get("choices");
                if (choices == null || !choices.isArray() || choices.isEmpty()) {
                    if (StrUtil.isNotBlank(directUrl)) {
                        finalContent = directUrl;
                    }
                    continue;
                }

                JsonNode choice = choices.get(0);
                String finishReason = textOf(choice.get("finish_reason"), null);
                JsonNode delta = choice.get("delta");
                JsonNode message = choice.get("message");
                String content = textOf(delta != null ? delta.get("content") : null, null);
                if (StrUtil.isBlank(content)) {
                    content = textOf(message != null ? message.get("content") : null, null);
                }
                if (StrUtil.isBlank(content) && StrUtil.isNotBlank(directUrl)) {
                    content = directUrl;
                }

                if ("stop".equalsIgnoreCase(finishReason) && StrUtil.isNotBlank(content)) {
                    finalContent = content;
                }
            } else if (line.startsWith("data:")) {
                eventLines.add(line.substring(5).trim());
            }
        }

        if (StrUtil.isBlank(finalContent)) {
            throw new BusinessException("GoogleFlowReverseApi 未返回最终结果");
        }
        return new CompletionResult(requestId, finalContent);
    }

    private String flushEvent(List<String> eventLines) {
        if (eventLines.isEmpty()) {
            return null;
        }
        String payload = String.join("\n", eventLines).trim();
        eventLines.clear();
        return payload;
    }

    private String toDataUrl(String sourceUrl) throws IOException {
        if (StrUtil.isBlank(sourceUrl)) {
            throw new BusinessException("GoogleFlowReverseApi 参考图地址为空");
        }
        String trimmed = sourceUrl.trim();
        if (trimmed.startsWith("data:")) {
            return trimmed;
        }

        BinaryResource resource = loadBinaryResource(trimmed);
        return "data:" + resource.mimeType() + ";base64," + Base64.getEncoder().encodeToString(resource.bytes());
    }

    private BinaryResource loadBinaryResource(String sourceUrl) throws IOException {
        if (sourceUrl.startsWith("/media/")) {
            return loadLocalMedia(sourceUrl);
        }
        Request request = new Request.Builder()
                .url(sourceUrl)
                .get()
                .addHeader("Accept", "image/*,*/*;q=0.8")
                .build();
        try (Response response = downloadHttpClient.newCall(request).execute()) {
            if (!response.isSuccessful() || response.body() == null) {
                throw new BusinessException("下载参考图失败: HTTP " + response.code() + " url=" + sourceUrl);
            }
            String mimeType = normalizeMimeType(response.header("Content-Type"), sourceUrl);
            return new BinaryResource(response.body().bytes(), mimeType);
        }
    }

    private BinaryResource loadLocalMedia(String sourceUrl) throws IOException {
        String relativePath = sourceUrl.replaceFirst("^/media/?", "");
        List<Path> candidates = new ArrayList<>();
        StorageConfig config = storageConfigService.getDefaultConfig();
        if (config != null && StrUtil.isNotBlank(config.getBasePath())) {
            candidates.add(Paths.get(config.getBasePath()).resolve(relativePath));
        }
        candidates.add(Paths.get(DEFAULT_LOCAL_MEDIA_BASE_PATH).resolve(relativePath));

        for (Path candidate : candidates) {
            if (candidate != null && Files.exists(candidate) && Files.isRegularFile(candidate)) {
                return new BinaryResource(Files.readAllBytes(candidate), normalizeMimeType(null, candidate.getFileName().toString()));
            }
        }
        throw new BusinessException("本地参考图不存在: " + sourceUrl);
    }

    private String normalizeMimeType(String contentType, String sourceUrl) {
        if (StrUtil.isNotBlank(contentType)) {
            return contentType.split(";", 2)[0].trim();
        }
        String lower = sourceUrl.toLowerCase(Locale.ROOT);
        if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) {
            return "image/jpeg";
        }
        if (lower.endsWith(".webp")) {
            return "image/webp";
        }
        if (lower.endsWith(".gif")) {
            return "image/gif";
        }
        return "image/png";
    }

    private String extractErrorMessage(String errorBody) {
        if (StrUtil.isBlank(errorBody)) {
            return "请求失败";
        }
        try {
            JsonNode root = OBJECT_MAPPER.readTree(errorBody);
            JsonNode error = root.get("error");
            if (error != null) {
                return textOf(error.get("message"), textOf(error.get("detail"), errorBody));
            }
            return textOf(root.get("detail"), errorBody);
        } catch (Exception ignored) {
            return errorBody;
        }
    }

    private String textOf(JsonNode node, String fallback) {
        if (node == null || node.isNull()) {
            return fallback;
        }
        String value = node.asText();
        return StrUtil.isBlank(value) ? fallback : value;
    }

    private record BinaryResource(byte[] bytes, String mimeType) {
    }

    public record CompletionResult(String requestId, String content) {
    }
}