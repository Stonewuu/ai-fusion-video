package com.stonewu.fusion.service.generation.strategy.impl;

import cn.hutool.core.codec.Base64;
import cn.hutool.core.util.StrUtil;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.auth.oauth2.GoogleCredentials;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.ImageItem;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import com.stonewu.fusion.service.generation.ImageGenerationService;
import com.stonewu.fusion.service.generation.strategy.ImageGenerationStrategy;
import com.stonewu.fusion.service.storage.MediaStorageService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import org.springframework.stereotype.Component;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Google Vertex AI (Imagen) 图片生成策略
 * <p>
 * 通过 Vertex AI REST API 调用 Imagen 模型进行文生图。
 * <p>
 * ApiConfig 字段映射：
 * - appId: Google Cloud 项目 ID (Project ID)
 * - apiUrl: location (如 us-central1)，也可填写完整 predict URL 覆盖默认地址
 * - appSecret: 服务账号 JSON Key 内容（完整 JSON 字符串）
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class VertexAiImageStrategy implements ImageGenerationStrategy {

    private static final String VERTEX_AI_SCOPE = "https://www.googleapis.com/auth/cloud-platform";
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    private final ImageGenerationService imageGenerationService;
    private final AiModelService aiModelService;
    private final MediaStorageService mediaStorageService;
    private final OkHttpClient okHttpClient = new OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(120, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            .build();

    @Override
    public String getName() {
        return "vertex_ai";
    }

    @Override
    public List<String> generate(String prompt, String modelCode, int width, int height, int count,
                                  List<String> imageUrls, ApiConfig apiConfig) {
        if (imageUrls != null && !imageUrls.isEmpty()) {
            throw new BusinessException("当前图片模型 " + modelCode + " 使用的是 Vertex AI Imagen 文生图接口，不支持参考图输入");
        }

        String projectId = resolveProjectId(apiConfig);
        String location = resolveLocation(apiConfig);

        // 构建请求 URL
        String url;
        if (isCustomPredictUrl(apiConfig.getApiUrl())) {
            url = apiConfig.getApiUrl();
        } else {
            url = String.format(
                    "https://%s-aiplatform.googleapis.com/v1/projects/%s/locations/%s/publishers/google/models/%s:predict",
                    location, projectId, location, modelCode
            );
        }

        // 构建请求体
        String aspectRatio = mapAspectRatio(width, height);
        String requestBody = buildRequestBody(prompt, count, aspectRatio);

        log.info("[VertexAI] 调用文生图 API: model={}, prompt={}, aspectRatio={}", modelCode, prompt, aspectRatio);

        try {
            String accessToken = getAccessToken(apiConfig);

            Request request = new Request.Builder()
                    .url(url)
                    .addHeader("Authorization", "Bearer " + accessToken)
                    .addHeader("Content-Type", "application/json")
                    .post(RequestBody.create(requestBody, MediaType.get("application/json")))
                    .build();

            OkHttpClient client = AiProxySupport.okHttpClient(okHttpClient, apiConfig);
            try (Response response = client.newCall(request).execute()) {
                if (!response.isSuccessful()) {
                    String errorBody = response.body() != null ? response.body().string() : "unknown";
                    throw new RuntimeException("Vertex AI 请求失败: HTTP " + response.code() + " - " + errorBody);
                }

                String responseBody = response.body().string();
                return parseImageUrls(responseBody);
            }
        } catch (IOException e) {
            throw new RuntimeException("Vertex AI 调用异常: " + e.getMessage(), e);
        }
    }

    @Override
    public String submit(ImageTask task, ApiConfig apiConfig) {
        AiModel model = resolveModel(task);
        String modelCode = (model != null && StrUtil.isNotBlank(model.getCode())) ? model.getCode() : "imagen-3.0-generate-002";
        int[] size = resolveDefaultSize(model, task);
        int count = (task.getCount() != null && task.getCount() > 0) ? task.getCount() : 1;

        // 解析参考图（图生图场景）
        List<String> imageUrls = parseRefImageUrls(task.getRefImageUrls());

        // 复用纯 API 调用
        List<String> urls = generate(task.getPrompt(), modelCode, size[0], size[1], count, imageUrls, apiConfig);

        // 更新数据库记录
        List<ImageItem> items = imageGenerationService.listItems(task.getId());
        int successCount = 0;
        for (int i = 0; i < urls.size() && i < items.size(); i++) {
            ImageItem item = items.get(i);
            item.setImageUrl(urls.get(i));
            item.setStatus(1);
            imageGenerationService.updateItem(item);
            successCount++;
        }

        task.setSuccessCount(successCount);
        imageGenerationService.update(task);

        log.info("[VertexAI] 文生图完成: taskId={}", task.getTaskId());
        return task.getTaskId();
    }

    @Override
    public void poll(String platformTaskId, ImageTask task, ApiConfig apiConfig) {
        // Vertex AI Imagen predict 是同步 API，submit 中已处理完成，无需轮询
    }

    /**
     * 构建 Vertex AI Imagen 请求体
     */
    private String buildRequestBody(String prompt, int sampleCount, String aspectRatio) {
        try {
            var mapper = OBJECT_MAPPER;
            var root = mapper.createObjectNode();

            var instances = mapper.createArrayNode();
            var instance = mapper.createObjectNode();
            instance.put("prompt", prompt);
            instances.add(instance);
            root.set("instances", instances);

            var parameters = mapper.createObjectNode();
            parameters.put("sampleCount", sampleCount);
            if (aspectRatio != null) {
                parameters.put("aspectRatio", aspectRatio);
            }
            root.set("parameters", parameters);

            return mapper.writeValueAsString(root);
        } catch (Exception e) {
            throw new RuntimeException("构建请求体失败: " + e.getMessage(), e);
        }
    }

    /**
     * 解析 Vertex AI 响应，提取图片 URL 列表
     */
    private List<String> parseImageUrls(String responseBody) {
        try {
            JsonNode root = OBJECT_MAPPER.readTree(responseBody);
            JsonNode predictions = root.get("predictions");

            if (predictions == null || !predictions.isArray() || predictions.isEmpty()) {
                throw new RuntimeException("Vertex AI 返回空结果");
            }

            List<String> urls = new ArrayList<>();
            for (JsonNode prediction : predictions) {
                if (prediction.has("bytesBase64Encoded")) {
                    String base64Data = prediction.get("bytesBase64Encoded").asText();
                    byte[] imageBytes = Base64.decode(base64Data);
                    // 直接通过 MediaStorageService 持久化，避免产生孤立临时文件
                    String persistedUrl = mediaStorageService.storeBytes(imageBytes, "images", "png");
                    urls.add(persistedUrl);
                } else if (prediction.has("gcsUri")) {
                    urls.add(prediction.get("gcsUri").asText());
                }
            }
            return urls;
        } catch (IOException e) {
            throw new RuntimeException("解析 Vertex AI 响应失败: " + e.getMessage(), e);
        }
    }

    /**
     * 获取 Google Cloud Access Token
     */
    private String getAccessToken(ApiConfig apiConfig) throws IOException {
        if (StrUtil.isBlank(apiConfig.getAppSecret())) {
            throw new RuntimeException("Vertex AI 配置缺少服务账号 JSON Key（appSecret 字段）");
        }

        var transportFactory = AiProxySupport.googleHttpTransportFactory(apiConfig);
        GoogleCredentials credentials = (transportFactory == null
            ? GoogleCredentials.fromStream(new ByteArrayInputStream(apiConfig.getAppSecret().getBytes(StandardCharsets.UTF_8)))
            : GoogleCredentials.fromStream(new ByteArrayInputStream(apiConfig.getAppSecret().getBytes(StandardCharsets.UTF_8)), transportFactory))
                .createScoped(Collections.singletonList(VERTEX_AI_SCOPE));
        credentials.refreshIfExpired();
        return credentials.getAccessToken().getTokenValue();
    }

    /**
     * 将像素尺寸映射为 Imagen 支持的 aspectRatio
     */
    private String mapAspectRatio(int width, int height) {
        double ratio = (double) width / height;
        if (Math.abs(ratio - 1.0) < 0.01) return "1:1";
        if (Math.abs(ratio - 16.0 / 9.0) < 0.05) return "16:9";
        if (Math.abs(ratio - 9.0 / 16.0) < 0.05) return "9:16";
        if (Math.abs(ratio - 4.0 / 3.0) < 0.05) return "4:3";
        if (Math.abs(ratio - 3.0 / 4.0) < 0.05) return "3:4";
        return "1:1";
    }

    private String resolveProjectId(ApiConfig apiConfig) {
        String projectId = apiConfig != null ? apiConfig.getAppId() : null;
        if (StrUtil.isNotBlank(projectId)) {
            return projectId;
        }
        // 兼容早期配置：曾使用 apiKey 存放 Project ID。
        projectId = apiConfig != null ? apiConfig.getApiKey() : null;
        if (StrUtil.isBlank(projectId)) {
            throw new BusinessException("Vertex AI 配置缺少 Project ID");
        }
        return projectId;
    }

    private String resolveLocation(ApiConfig apiConfig) {
        String location = apiConfig != null ? apiConfig.getApiUrl() : null;
        if (StrUtil.isNotBlank(location) && !isCustomPredictUrl(location)) {
            return location;
        }
        // 兼容早期配置：当 apiKey 存 Project ID 时，appId 可能存放 location。
        if (apiConfig != null && StrUtil.isBlank(apiConfig.getApiUrl())
                && StrUtil.isNotBlank(apiConfig.getApiKey()) && StrUtil.isNotBlank(apiConfig.getAppId())) {
            return apiConfig.getAppId();
        }
        return "us-central1";
    }

    private boolean isCustomPredictUrl(String apiUrl) {
        return StrUtil.startWithIgnoreCase(apiUrl, "http://")
                || StrUtil.startWithIgnoreCase(apiUrl, "https://");
    }

    private AiModel resolveModel(ImageTask task) {
        if (task.getModelId() != null) {
            AiModel model = aiModelService.getById(task.getModelId());
            if (model != null && StrUtil.isNotBlank(model.getCode())) {
                return model;
            }
        }
        return null;
    }
}
