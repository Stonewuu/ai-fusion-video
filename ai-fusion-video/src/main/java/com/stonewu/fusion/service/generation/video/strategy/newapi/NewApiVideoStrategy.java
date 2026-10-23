package com.stonewu.fusion.service.generation.video.strategy.newapi;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.model.AiModelMetadata;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import com.stonewu.fusion.service.generation.GenerationModelCapabilityService;
import com.stonewu.fusion.service.generation.job.*;
import com.stonewu.fusion.service.generation.transport.GenerationHttpClients;
import com.stonewu.fusion.service.generation.video.VideoGenerationService;
import com.stonewu.fusion.service.generation.video.strategy.VideoGenerationStrategy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import okhttp3.*;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;

/**
 * New API 视频生成策略。
 * <p>
 * 对接官方通用视频接口：
 * POST /v1/video/generations
 * GET /v1/video/generations/{task_id}
 * <p>
 * 当前实现按官方通用字段适配为：文生视频 / 单图生视频。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class NewApiVideoStrategy implements VideoGenerationStrategy {

    public static final String PLATFORM = "newapi";

    private static final String DEFAULT_BASE_URL = "https://docs.newapi.ai";
    private static final long DEFAULT_POLL_INTERVAL_MILLIS = 10000L;
    private static final long DEFAULT_POLL_TIMEOUT_MILLIS = 30L * 60L * 1000L;
    private static final MediaType JSON_MEDIA_TYPE = MediaType.get("application/json");

    private final AiModelService aiModelService;
    private final ApiConfigService apiConfigService;
    private final VideoGenerationService videoGenerationService;
    private final AiModelMetadataResolver aiModelMetadataResolver;
    private final GenerationModelCapabilityService generationModelCapabilityService;
    private final NewApiVideoProtocolRouter protocolRouter;

    private final OkHttpClient okHttpClient = GenerationHttpClients.providerRequestBuilder()
            .connectTimeout(1, TimeUnit.MINUTES)
            .readTimeout(25, TimeUnit.MINUTES)
            .writeTimeout(30, TimeUnit.SECONDS)
            .build();

    @Override
    public String getName() {
        return PLATFORM;
    }

    @Override
    public Set<String> getSupportedProtocols() {
        return protocolRouter.getSupportedProtocols();
    }

    @Override
    public PreparedGenerationCall prepare(VideoTask task, AiModel model, ApiConfig apiConfig) {
        var config = generationModelCapabilityService.getMergedModelConfig(model);
        var context = buildProtocolContext(model, apiConfig, task, config);
        var adapter = protocolRouter.resolve(context);
        Request request = new Request.Builder().url(resolveApiUrl(apiConfig, adapter.resolveSubmitPath(context)))
                .addHeader("Authorization", "Bearer " + apiConfig.getApiKey())
                .addHeader("Content-Type", "application/json")
                .post(RequestBody.create(adapter.buildSubmitBody(context).toString(), JSON_MEDIA_TYPE)).build();
        var client = AiProxySupport.okHttpClient(okHttpClient, apiConfig);
        return new PreparedGenerationCall(GenerationPreparedRequest.ordinary(), () ->
                GenerationProviderReceipt.accepted(submitTask(client, request)));
    }

    @Override
    public RemoteGenerationResult query(String platformTaskId, VideoTask task, AiModel model, ApiConfig apiConfig) {
        var config = generationModelCapabilityService.getMergedModelConfig(model);
        var context = buildProtocolContext(model, apiConfig, task, config);
        var adapter = protocolRouter.resolve(context);
        var result = queryTask(apiConfig, adapter.resolveQueryPath(context, platformTaskId));
        Long interval = getPositiveLong(config, "pollIntervalMillis", "pollIntervalMs", "pollInterval");
        return ProviderJobStates.result(normalizeStatus(result.status()), result.errorMessage(),
                interval == null ? 10 : (int) Math.ceil(interval / 1000.0),
                () -> List.of(new RemoteGenerationResult.Output(
                        result.videoUrl(), result.coverUrl(), result.firstFrameUrl(), result.lastFrameUrl(),
                        result.duration() == null ? task.getDuration() : result.duration())));
    }

    private String submitTask(OkHttpClient client, Request request) {
        try (Response response = client.newCall(request).execute()) {
            String responseBody = response.body() != null ? response.body().string() : "";
            if (!response.isSuccessful()) {
                throw new BusinessException("New API 视频任务提交失败: HTTP " + response.code() + " "
                        + extractErrorMessage(responseBody));
            }
            String taskId = extractTaskId(responseBody);
            if (StrUtil.isBlank(taskId)) {
                throw new BusinessException("New API 视频任务未返回 task_id");
            }
            return taskId;
        } catch (IOException e) {
            throw new BusinessException("New API 视频任务提交异常: " + e.getMessage());
        }
    }


    private NewApiVideoResult queryTask(ApiConfig apiConfig, String queryPath) {
        Request request = new Request.Builder()
                .url(resolveApiUrl(apiConfig, queryPath))
                .addHeader("Authorization", "Bearer " + apiConfig.getApiKey())
                .get()
                .build();

        OkHttpClient client = AiProxySupport.okHttpClient(okHttpClient, apiConfig);
        try (Response response = client.newCall(request).execute()) {
            String responseBody = response.body() != null ? response.body().string() : "";
            if (!response.isSuccessful()) {
                throw new BusinessException("New API 视频任务查询失败: HTTP " + response.code() + " "
                        + extractErrorMessage(responseBody));
            }
            return parseQueryResult(responseBody);
        } catch (IOException e) {
            throw new BusinessException("New API 视频任务查询异常: " + e.getMessage());
        }
    }

    private NewApiVideoResult parseQueryResult(String responseBody) {
        JSONObject root = parseObject(responseBody, "New API 视频任务查询响应不是合法 JSON");
        JSONObject metadata = firstObject(root.getJSONObject("metadata"),
                objectField(root, "output", "metadata"),
                objectField(root, "result", "metadata"),
                arrayObjectField(root, "data", 0, "metadata"));

        String status = firstNonBlank(
                root.getStr("status"),
                objectFieldString(root, "output", "status"),
                objectFieldString(root, "result", "status"),
                arrayObjectFieldString(root, "data", 0, "status"));

        String videoUrl = firstNonBlank(
                root.getStr("url"),
                root.getStr("video_url"),
                objectFieldString(root, "output", "url"),
                objectFieldString(root, "output", "video_url"),
                objectFieldString(root, "result", "url"),
                objectFieldString(root, "result", "video_url"),
                arrayObjectFieldString(root, "data", 0, "url"),
                arrayObjectFieldString(root, "data", 0, "video_url"));

        String coverUrl = firstNonBlank(
                root.getStr("cover_url"),
                root.getStr("coverUrl"),
                objectFieldString(root, "output", "cover_url"),
                objectFieldString(root, "output", "coverUrl"),
                objectFieldString(root, "result", "cover_url"),
                objectFieldString(root, "result", "coverUrl"),
                arrayObjectFieldString(root, "data", 0, "cover_url"),
                arrayObjectFieldString(root, "data", 0, "coverUrl"));

        String firstFrameUrl = firstNonBlank(
                root.getStr("first_frame_url"),
                root.getStr("firstFrameUrl"),
                objectFieldString(root, "output", "first_frame_url"),
                objectFieldString(root, "output", "firstFrameUrl"),
                objectFieldString(root, "result", "first_frame_url"),
                objectFieldString(root, "result", "firstFrameUrl"));

        String lastFrameUrl = firstNonBlank(
                root.getStr("last_frame_url"),
                root.getStr("lastFrameUrl"),
                objectFieldString(root, "output", "last_frame_url"),
                objectFieldString(root, "output", "lastFrameUrl"),
                objectFieldString(root, "result", "last_frame_url"),
                objectFieldString(root, "result", "lastFrameUrl"));

        Integer duration = firstPositive(
                root.getInt("duration"),
                metadata != null ? metadata.getInt("duration") : null,
                objectField(root, "output", "metadata") != null
                        ? objectField(root, "output", "metadata").getInt("duration") : null,
                objectField(root, "result", "metadata") != null
                        ? objectField(root, "result", "metadata").getInt("duration") : null);

        String errorMessage = firstNonBlank(
                extractErrorMessage(root.get("error")),
                root.getStr("message"),
                root.getStr("detail"));

        return new NewApiVideoResult(status, videoUrl, coverUrl, firstFrameUrl, lastFrameUrl, duration, errorMessage);
    }

    private String extractTaskId(String responseBody) {
        JSONObject root = parseObject(responseBody, "New API 视频任务提交响应不是合法 JSON");
        return firstNonBlank(
                root.getStr("task_id"),
                root.getStr("id"),
                objectFieldString(root, "data", "task_id"),
                objectFieldString(root, "data", "id"));
    }


    private String resolveApiUrl(ApiConfig apiConfig, String path) {
        String normalizedPath = StrUtil.blankToDefault(path, NewApiVideoProtocolAdapter.DEFAULT_SUBMIT_PATH);
        if (!normalizedPath.startsWith("/")) {
            normalizedPath = "/" + normalizedPath;
        }
        return normalizeRootBaseUrl(apiConfig.getApiUrl()) + normalizedPath;
    }

    private NewApiVideoProtocolContext buildProtocolContext(AiModel model, ApiConfig apiConfig,
                                                            VideoTask task, JSONObject modelConfig) {
        AiModelMetadata metadata = aiModelMetadataResolver.resolve(model, apiConfig);
        return new NewApiVideoProtocolContext(model, apiConfig, task, modelConfig, metadata);
    }

    private String normalizeRootBaseUrl(String baseUrl) {
        String normalized = StrUtil.blankToDefault(StrUtil.trim(baseUrl), DEFAULT_BASE_URL).replaceAll("/+$", "");
        if (normalized.endsWith("/v1")) {
            normalized = normalized.substring(0, normalized.length() - 3);
        }
        return normalized;
    }

    private String extractErrorMessage(String responseBody) {
        if (StrUtil.isBlank(responseBody)) {
            return "响应体为空";
        }
        try {
            JSONObject root = JSONUtil.parseObj(responseBody);
            String message = firstNonBlank(
                    extractErrorMessage(root.get("error")),
                    root.getStr("message"),
                    root.getStr("detail"));
            return StrUtil.blankToDefault(message, preview(responseBody));
        } catch (Exception ignored) {
            return preview(responseBody);
        }
    }

    private String extractErrorMessage(Object errorObject) {
        if (errorObject == null) {
            return null;
        }
        if (errorObject instanceof JSONObject errorJson) {
            return firstNonBlank(errorJson.getStr("message"), errorJson.getStr("detail"), errorJson.getStr("code"));
        }
        return errorObject.toString();
    }

    private String preview(String value) {
        if (StrUtil.isBlank(value)) {
            return "";
        }
        String normalized = value.replaceAll("\\s+", " ").trim();
        return normalized.length() > 200 ? normalized.substring(0, 200) + "..." : normalized;
    }

    private JSONObject parseObject(String json, String errorMessage) {
        try {
            return JSONUtil.parseObj(json);
        } catch (Exception e) {
            throw new BusinessException(errorMessage);
        }
    }

    private JSONObject objectField(JSONObject root, String field, String nestedField) {
        if (root == null) {
            return null;
        }
        Object value = root.get(field);
        if (!(value instanceof JSONObject jsonObject)) {
            return null;
        }
        Object nested = jsonObject.get(nestedField);
        return nested instanceof JSONObject nestedObject ? nestedObject : null;
    }

    private String objectFieldString(JSONObject root, String field, String nestedField) {
        if (root == null) {
            return null;
        }
        Object value = root.get(field);
        if (!(value instanceof JSONObject jsonObject)) {
            return null;
        }
        return jsonObject.getStr(nestedField);
    }

    private JSONObject arrayObjectField(JSONObject root, String arrayField, int index, String nestedField) {
        if (root == null) {
            return null;
        }
        Object value = root.get(arrayField);
        if (!(value instanceof JSONArray array) || array.size() <= index) {
            return null;
        }
        Object item = array.get(index);
        if (!(item instanceof JSONObject jsonObject)) {
            return null;
        }
        Object nested = jsonObject.get(nestedField);
        return nested instanceof JSONObject nestedObject ? nestedObject : null;
    }

    private String arrayObjectFieldString(JSONObject root, String arrayField, int index, String nestedField) {
        if (root == null) {
            return null;
        }
        Object value = root.get(arrayField);
        if (!(value instanceof JSONArray array) || array.size() <= index) {
            return null;
        }
        Object item = array.get(index);
        if (!(item instanceof JSONObject jsonObject)) {
            return null;
        }
        return jsonObject.getStr(nestedField);
    }

    private JSONObject firstObject(JSONObject... candidates) {
        for (JSONObject candidate : candidates) {
            if (candidate != null && !candidate.isEmpty()) {
                return candidate;
            }
        }
        return null;
    }

    private Integer getPositiveInteger(JSONObject config, String... keys) {
        if (config == null) {
            return null;
        }
        for (String key : keys) {
            Object value = config.get(key);
            if (value == null) {
                continue;
            }
            try {
                int parsed = value instanceof Number number ? number.intValue() : Integer.parseInt(
                        value.toString().trim());
                if (parsed > 0) {
                    return parsed;
                }
            } catch (Exception ignored) {
                return null;
            }
        }
        return null;
    }

    private Long getPositiveLong(JSONObject config, String... keys) {
        if (config == null) {
            return null;
        }
        for (String key : keys) {
            Object value = config.get(key);
            if (value == null) {
                continue;
            }
            try {
                long parsed = value instanceof Number number ? number.longValue() : Long.parseLong(
                        value.toString().trim());
                if (parsed > 0) {
                    return parsed;
                }
            } catch (Exception ignored) {
                return null;
            }
        }
        return null;
    }

    private Integer firstPositive(Integer... values) {
        for (Integer value : values) {
            if (value != null && value > 0) {
                return value;
            }
        }
        return null;
    }

    private String firstNonBlank(String... values) {
        for (String value : values) {
            if (StrUtil.isNotBlank(value)) {
                return value;
            }
        }
        return null;
    }

    private String normalizeStatus(String status) {
        return status == null ? "" : status.trim().toLowerCase();
    }


    private record NewApiVideoResult(String status, String videoUrl, String coverUrl,
                                     String firstFrameUrl, String lastFrameUrl,
                                     Integer duration, String errorMessage) {
    }
}
