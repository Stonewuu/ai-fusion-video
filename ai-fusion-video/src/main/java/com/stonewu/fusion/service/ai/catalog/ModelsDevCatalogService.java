package com.stonewu.fusion.service.ai.catalog;

import com.fasterxml.jackson.databind.JsonNode;
import com.stonewu.fusion.service.ai.google.GoogleModelFamily;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import java.io.IOException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.concurrent.TimeUnit;

/** 缓存公开模型目录；用户的 API 密钥不会发送给 Models.dev。 */
@Service
@RequiredArgsConstructor
@Slf4j
public class ModelsDevCatalogService {
    public static final String SOURCE = "https://models.dev/api.json";
    private static final int MAX_BYTES = 32 * 1024 * 1024;
    private static final OkHttpClient HTTP = new OkHttpClient.Builder()
            .connectTimeout(10, TimeUnit.SECONDS).readTimeout(25, TimeUnit.SECONDS)
            .callTimeout(35, TimeUnit.SECONDS).followRedirects(false).build();
    private final ApiConfigService apiConfigs;
    private final ObjectMapper json;

    @Cacheable(value = "modelsDevCatalog", key = "'generation-v3'", sync = true)
    @Transactional(propagation = Propagation.NEVER)
    public Snapshot load(Long apiConfigId) {
        var api = apiConfigId == null ? null : apiConfigs.getById(apiConfigId);
        if (apiConfigId != null && api == null) {
            throw new BusinessException("API 配置不存在，请重新选择");
        }
        var client = AiProxySupport.okHttpClient(HTTP, api);
        try (var response = client.newCall(new Request.Builder().url(SOURCE).get().build()).execute()) {
            if (!response.isSuccessful() || response.body() == null) {
                log.warn("Models.dev 目录请求失败: HTTP {}", response.code());
                throw new BusinessException("暂时无法获取模型目录，请重试");
            }
            byte[] bytes = response.body().byteStream().readNBytes(MAX_BYTES + 1);
            if (bytes.length > MAX_BYTES) {
                throw new BusinessException("模型目录暂时无法加载，请稍后重试");
            }
            return parse(json.readTree(bytes));
        } catch (IOException failure) {
            log.warn("Models.dev 连接失败: {}", failure.toString());
            throw new BusinessException("Models.dev 暂时无法连接，请重试");
        }
    }

    @CacheEvict(value = "modelsDevCatalog", allEntries = true)
    public void refresh() {
    }

    static Snapshot parse(JsonNode root) {
        if (!root.isObject()) {
            throw new BusinessException("模型目录暂时无法加载，请稍后重试");
        }
        var entries = new ArrayList<Entry>();
        root.fields().forEachRemaining(provider -> {
            JsonNode models = provider.getValue().path("models");
            if (!models.isObject()) {
                return;
            }
            models.fields().forEachRemaining(model -> {
                JsonNode data = model.getValue();
                List<String> outputs = strings(data.path("modalities").path("output"));
                Integer type = modelType(model.getKey(), data.path("type").asText(), outputs);
                if (type == null || model.getKey().isBlank()) {
                    return;
                }
                List<String> inputs = strings(data.path("modalities").path("input"));
                List<String> levels = null;
                if (data.path("reasoning_options").isArray()) {
                    var efforts = new LinkedHashSet<String>();
                    for (JsonNode option : data.path("reasoning_options")) {
                        if ("effort".equals(option.path("type").asText()) && option.path("values").isArray()) {
                            efforts.addAll(strings(option.path("values")));
                        }
                    }
                    levels = List.copyOf(efforts);
                }
                entries.add(new Entry(provider.getKey(), provider.getValue().path("name").asText(provider.getKey()),
                        model.getKey(), data.path("name").asText(model.getKey()),
                        data.path("canonical_model_id").asText(null), type,
                        positiveInteger(data.path("limit").path("context")),
                        positiveInteger(data.path("limit").path("output")), inputs, outputs,
                        data.path("reasoning").isBoolean() ? data.path("reasoning").asBoolean() : null,
                        levels, data.path("tool_call").isBoolean() ? data.path("tool_call").asBoolean() : null,
                        data.path("last_updated").asText(null), data.path("status").asText(null)));
            });
        });
        if (entries.isEmpty()) {
            throw new BusinessException("模型目录暂时无法加载，请稍后重试");
        }
        return new Snapshot(Instant.now().toString(), List.copyOf(entries));
    }

    private static Integer modelType(String code, String type, List<String> output) {
        String id = code.substring(code.lastIndexOf('/') + 1);
        if (GoogleModelFamily.excluded(id) || "embedding".equals(type) || "transcription".equals(type)) {
            return null;
        }
        Integer googleType = GoogleModelFamily.type(id);
        if (googleType != null) {
            return googleType;
        }
        if (output == null || "decision".equals(type)) {
            return null;
        }
        if (output.contains("video")) {
            return 3;
        }
        if (output.contains("image")) {
            return 2;
        }
        if (output.contains("audio")) {
            return 4;
        }
        return output.contains("text") ? 1 : null;
    }

    private static Integer positiveInteger(JsonNode value) {
        return value.canConvertToInt() && value.isIntegralNumber() && value.intValue() > 0 ? value.intValue() : null;
    }

    private static List<String> strings(JsonNode value) {
        if (!value.isArray()) {
            return null;
        }
        var result = new LinkedHashSet<String>();
        for (JsonNode item : value) {
            if (item.isTextual() && !item.asText().isBlank()) {
                result.add(item.asText());
            }
        }
        return List.copyOf(result);
    }

    public record Snapshot(String fetchedAt, List<Entry> models) {
    }

    public record Entry(String providerId, String providerName, String modelId, String name,
                        String canonicalModelId, int modelType, Integer contextWindow, Integer outputTokenLimit,
                        List<String> inputModalities, List<String> outputModalities, Boolean supportReasoning,
                        List<String> reasoningEffortLevels, Boolean toolCall, String updatedAt, String status) {
    }
}
