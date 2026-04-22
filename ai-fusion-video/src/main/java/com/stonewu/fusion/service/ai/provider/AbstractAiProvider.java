package com.stonewu.fusion.service.ai.provider;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.controller.ai.vo.RemoteModelVO;
import lombok.extern.slf4j.Slf4j;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/**
 * 提供商公共能力基类。
 */
@Slf4j
public abstract class AbstractAiProvider implements AiProvider {

    private static final int RESPONSE_PREVIEW_LENGTH = 180;

    private final OkHttpClient httpClient = new OkHttpClient.Builder()
            .connectTimeout(15, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .build();

    protected String normalizeBaseUrl(String baseUrl) {
        if (StrUtil.isBlank(baseUrl)) {
            return baseUrl;
        }
        return baseUrl.replaceAll("/+$", "");
    }

    protected String joinUrl(String baseUrl, String path) {
        String normalizedBaseUrl = normalizeBaseUrl(baseUrl);
        if (StrUtil.isBlank(normalizedBaseUrl)) {
            return path;
        }
        if (StrUtil.isBlank(path) || "/".equals(path)) {
            return normalizedBaseUrl;
        }
        return normalizedBaseUrl + (path.startsWith("/") ? path : "/" + path);
    }

    protected String ensurePathSuffix(String baseUrl, String suffix) {
        String normalizedBaseUrl = normalizeBaseUrl(baseUrl);
        String normalizedSuffix = suffix.startsWith("/") ? suffix : "/" + suffix;
        if (StrUtil.isBlank(normalizedBaseUrl)) {
            return normalizedSuffix;
        }
        if (normalizedBaseUrl.equalsIgnoreCase(normalizedSuffix)
                || normalizedBaseUrl.toLowerCase().endsWith(normalizedSuffix.toLowerCase())) {
            return normalizedBaseUrl;
        }
        return normalizedBaseUrl + normalizedSuffix;
    }

    protected void requireApiKey(String apiKey, String platformName) {
        if (StrUtil.isBlank(apiKey)) {
            throw new BusinessException(platformName + " 模型缺少 apiKey 配置");
        }
    }

    protected String getStr(Map<String, Object> map, String key, String defaultValue) {
        Object value = map.get(key);
        return value != null ? value.toString() : defaultValue;
    }

    protected double toDouble(Object value) {
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        return Double.parseDouble(value.toString());
    }

    protected int toInt(Object value) {
        if (value instanceof Number number) {
            return number.intValue();
        }
        return Integer.parseInt(value.toString());
    }

    protected void applyDouble(Map<String, Object> config, String key, Consumer<Double> setter) {
        Object value = config.get(key);
        if (value == null) {
            return;
        }
        try {
            setter.accept(toDouble(value));
        } catch (Exception e) {
            log.warn("[AiProvider] 参数设置失败: key={}, value={}", key, value);
        }
    }

    protected void applyInt(Map<String, Object> config, String key, Consumer<Integer> setter) {
        Object value = config.get(key);
        if (value == null) {
            return;
        }
        try {
            setter.accept(toInt(value));
        } catch (Exception e) {
            log.warn("[AiProvider] 参数设置失败: key={}, value={}", key, value);
        }
    }

    protected Object getConfigValue(Map<String, Object> config, String... keys) {
        if (config == null || config.isEmpty()) {
            return null;
        }
        for (String key : keys) {
            if (config.containsKey(key)) {
                return config.get(key);
            }
        }
        return null;
    }

    protected String getConfigString(Map<String, Object> config, String... keys) {
        Object value = getConfigValue(config, keys);
        if (value == null) {
            return null;
        }
        String text = value.toString().trim();
        return text.isEmpty() ? null : text;
    }

    protected Integer getConfigInteger(Map<String, Object> config, String... keys) {
        Object value = getConfigValue(config, keys);
        if (value == null) {
            return null;
        }
        try {
            return toInt(value);
        } catch (Exception e) {
            log.warn("[AiProvider] 参数解析失败: keys={}, value={}", String.join(",", keys), value);
            return null;
        }
    }

    protected Boolean getConfigBoolean(Map<String, Object> config, String... keys) {
        Object value = getConfigValue(config, keys);
        if (value == null) {
            return null;
        }
        if (value instanceof Boolean bool) {
            return bool;
        }
        if (value instanceof Number number) {
            return number.intValue() != 0;
        }
        String text = value.toString().trim();
        if (text.isEmpty()) {
            return null;
        }
        if ("true".equalsIgnoreCase(text) || "1".equals(text)
                || "yes".equalsIgnoreCase(text) || "enabled".equalsIgnoreCase(text)) {
            return true;
        }
        if ("false".equalsIgnoreCase(text) || "0".equals(text)
                || "no".equalsIgnoreCase(text) || "disabled".equalsIgnoreCase(text)) {
            return false;
        }
        return null;
    }

    protected boolean isReasoningEnabled(AiProviderContext context) {
        if (context == null) {
            return false;
        }
        if (context.getModel() != null && Boolean.TRUE.equals(context.getModel().getSupportReasoning())) {
            return true;
        }
        Map<String, Object> config = context.getConfig();
        if (config == null || config.isEmpty()) {
            return false;
        }
        if (Boolean.TRUE.equals(getConfigBoolean(config,
                "enableThinking", "enable_thinking", "includeReasoning", "include_reasoning"))) {
            return true;
        }
        return getConfigInteger(config, "thinkingBudget", "thinking_budget") != null
                || StrUtil.isNotBlank(getConfigString(config, "reasoningEffort", "reasoning_effort"))
                || getConfigValue(config, "thinking") != null;
    }

    protected String executeGet(String url, Map<String, String> headers) {
        Request.Builder builder = new Request.Builder().url(url).get();
        headers.forEach(builder::addHeader);

        try (Response response = httpClient.newCall(builder.build()).execute()) {
            if (!response.isSuccessful()) {
                String body = response.body() != null ? response.body().string() : "";
                log.error("[AiProvider] 请求失败: url={}, code={}, message={}, body={}",
                        url, response.code(), response.message(), body);
                throw new BusinessException("获取模型列表失败: HTTP " + response.code() + " " + response.message());
            }
            if (response.body() == null) {
                throw new BusinessException("获取模型列表失败: 响应体为空");
            }
            return response.body().string();
        } catch (IOException e) {
            log.error("[AiProvider] 请求异常: url={}", url, e);
            throw new BusinessException("获取模型列表失败: " + e.getMessage());
        }
    }

    protected List<RemoteModelVO> parseDataArrayModels(String json, String ownerFallback) {
        List<RemoteModelVO> result = new ArrayList<>();
        try {
            JSONObject root = JSONUtil.parseObj(json);
            JSONArray data = root.getJSONArray("data");
            if (data == null) {
                return result;
            }
            for (int index = 0; index < data.size(); index++) {
                JSONObject item = data.getJSONObject(index);
                String id = item.getStr("id");
                if (StrUtil.isBlank(id)) {
                    continue;
                }
                String owner = item.getStr("owned_by");
                if (StrUtil.isBlank(owner)) {
                    owner = item.getStr("display_name", ownerFallback);
                }
                result.add(RemoteModelVO.builder()
                        .id(id)
                        .ownedBy(StrUtil.blankToDefault(owner, ownerFallback))
                        .build());
            }
        } catch (Exception e) {
            String responsePreview = previewResponse(json);
            log.error("[AiProvider] 解析模型列表失败, responsePreview={}", responsePreview, e);
            throw new BusinessException("解析模型列表响应失败，接口返回可能不是 JSON。响应预览: " + responsePreview);
        }
        result.sort(Comparator.comparing(RemoteModelVO::getId));
        return result;
    }

    protected List<RemoteModelVO> parseOllamaTags(String json) {
        List<RemoteModelVO> result = new ArrayList<>();
        try {
            JSONObject root = JSONUtil.parseObj(json);
            JSONArray models = root.getJSONArray("models");
            if (models == null) {
                return result;
            }
            for (int index = 0; index < models.size(); index++) {
                JSONObject item = models.getJSONObject(index);
                String id = item.getStr("name", item.getStr("model", ""));
                if (StrUtil.isBlank(id)) {
                    continue;
                }
                result.add(RemoteModelVO.builder()
                        .id(id)
                        .ownedBy("ollama")
                        .build());
            }
        } catch (Exception e) {
            String responsePreview = previewResponse(json);
            log.error("[AiProvider] 解析 Ollama 模型列表失败, responsePreview={}", responsePreview, e);
            throw new BusinessException("解析模型列表响应失败，接口返回可能不是 JSON。响应预览: " + responsePreview);
        }
        result.sort(Comparator.comparing(RemoteModelVO::getId));
        return result;
    }

    protected List<RemoteModelVO> parseGeminiModels(String json) {
        Map<String, RemoteModelVO> deduplicated = new LinkedHashMap<>();
        try {
            JSONObject root = JSONUtil.parseObj(json);
            JSONArray models = root.getJSONArray("models");
            if (models == null) {
                return List.of();
            }
            for (int index = 0; index < models.size(); index++) {
                JSONObject item = models.getJSONObject(index);
                if (!supportsGeminiGenerateContent(item.getJSONArray("supportedGenerationMethods"))) {
                    continue;
                }

                String id = item.getStr("baseModelId");
                if (StrUtil.isBlank(id)) {
                    id = StrUtil.removePrefix(item.getStr("name"), "models/");
                }
                if (StrUtil.isBlank(id)) {
                    continue;
                }

                String displayName = item.getStr("displayName");
                deduplicated.putIfAbsent(id, RemoteModelVO.builder()
                        .id(id)
                        .ownedBy(StrUtil.blankToDefault(displayName, "google"))
                        .modelType(1)
                        .build());
            }
        } catch (Exception e) {
            String responsePreview = previewResponse(json);
            log.error("[AiProvider] 解析 Gemini 模型列表失败, responsePreview={}", responsePreview, e);
            throw new BusinessException("解析 Gemini 模型列表响应失败。响应预览: " + responsePreview);
        }
        List<RemoteModelVO> result = new ArrayList<>(deduplicated.values());
        result.sort(Comparator.comparing(RemoteModelVO::getId));
        return result;
    }

    private boolean supportsGeminiGenerateContent(JSONArray supportedMethods) {
        if (supportedMethods == null) {
            return false;
        }
        for (int index = 0; index < supportedMethods.size(); index++) {
            String method = supportedMethods.getStr(index);
            if ("generateContent".equalsIgnoreCase(method)) {
                return true;
            }
        }
        return false;
    }

    private String previewResponse(String responseBody) {
        if (StrUtil.isBlank(responseBody)) {
            return "<empty>";
        }
        String compact = responseBody.replaceAll("\\s+", " ").trim();
        if (compact.length() <= RESPONSE_PREVIEW_LENGTH) {
            return compact;
        }
        return compact.substring(0, RESPONSE_PREVIEW_LENGTH) + "...";
    }
}