package com.stonewu.fusion.service.ai.catalog;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.ModelPresetService;
import com.stonewu.fusion.service.generation.transport.OfficialVideoHttpClient;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** 当前密钥的精确模型 ID 与模型广场明确声明的接口取交集，不按名称猜测类型。 */
@Component
@RequiredArgsConstructor
public class InfistarModelDiscovery implements ProviderModelDiscovery {
    private final ModelPresetService presets;

    @Override
    public String type() {
        return "infistar";
    }

    @Override
    public ProviderDiscoveryResult discover(ApiConfig config, ProviderDefinition provider) {
        var client = new OfficialVideoHttpClient(config);
        String v1 = OfficialVideoHttpClient.baseUrl(config, "/v1");
        var authorized = client.get(v1 + "/models");
        JSONArray available = requireArray(authorized, "data", "模型列表");
        var metadata = client.get(v1.substring(0, v1.length() - 3) + "/api/pricing");
        if (!Boolean.TRUE.equals(metadata.getBool("success"))) {
            throw new BusinessException("无限星河模型广场读取失败，请稍后重试");
        }
        JSONArray catalogRows = requireArray(metadata, "data", "模型广场");
        var endpoints = metadata.getJSONObject("supported_endpoint");
        if (endpoints == null) {
            throw new BusinessException("无限星河模型广场缺少接口说明，无法确认模型类型");
        }
        var catalog = new HashMap<String, JSONObject>();
        for (int index = 0; index < catalogRows.size(); index++) {
            JSONObject entry = catalogRows.getJSONObject(index);
            if (entry == null || StrUtil.isBlank(entry.getStr("model_name"))
                    || entry.getJSONArray("supported_endpoint_types") == null) {
                throw new BusinessException("无限星河模型广场缺少模型标识或接口类型，无法确认可用模型");
            }
            catalog.put(entry.getStr("model_name"), entry);
        }
        var candidates = new ArrayList<ProviderModelCandidate>();
        var seen = new HashSet<String>();
        for (int index = 0; index < available.size(); index++) {
            JSONObject model = available.getJSONObject(index);
            String id = model.getStr("id");
            if (StrUtil.isBlank(id)) {
                throw new BusinessException("无限星河模型列表包含空模型代码");
            }
            if (!seen.add(id)) {
                continue;
            }
            JSONObject entry = catalog.get(id);
            if (entry == null) {
                continue;
            }
            Set<String> paths = endpointPaths(entry.getJSONArray("supported_endpoint_types"), endpoints);
            JSONArray accountEndpoints = model.getJSONArray("supported_endpoint_types");
            if (accountEndpoints != null) {
                paths.retainAll(endpointPaths(accountEndpoints, endpoints));
            }
            ProviderModelCandidate candidate = candidate(id, paths, entry.getStr("tags"));
            if (candidate != null) {
                candidates.add(candidate);
            }
        }
        return new ProviderDiscoveryResult(provider.id(), true,
                "已按当前密钥和平台接口说明筛选对话、图片、视频模型。", List.copyOf(candidates));
    }

    private ProviderModelCandidate candidate(String id, Set<String> paths, String tags) {
        int type;
        String protocol;
        var overrides = new JSONObject();
        if (paths.contains("/v1/images/generations") || paths.contains("/v1/images/edits")) {
            type = 2;
            protocol = "openai";
            boolean edits = paths.contains("/v1/images/edits");
            overrides.set("supportReferenceImages", edits).set("maxReferenceImages", edits ? 1 : 0)
                    .set("referenceImageInputFormats", List.of("data_uri"));
        } else if (paths.contains("/v1/videos")) {
            type = 3;
            protocol = "infistar_video";
            overrides.set("supportFirstFrame", true).set("supportLastFrame", false)
                    .set("supportReferenceImages", false).set("supportReferenceVideos", false)
                    .set("supportReferenceAudios", false).set("maxImageInputs", 1)
                    .set("maxReferenceImages", 0).set("maxReferenceVideos", 0).set("maxReferenceAudios", 0)
                    .set("referenceImageInputFormats", List.of("url", "data_uri"))
                    .set("supportGenerateAudio", false).set("supportWatermark", false)
                    .set("supportSeed", false).set("supportCameraFixed", false);
        } else if (hasMediaGenerationTag(tags)) {
            // 平台部分图片模型借用对话端点，不能据此把图片输出接到文本执行链路。
            return null;
        } else if (paths.contains("/v1/chat/completions") || paths.contains("/v1/responses")) {
            type = 1;
            protocol = "openai_compatible";
            overrides.set("apiMode", paths.contains("/v1/chat/completions") ? "chat_completions" : "responses");
        } else if (paths.contains("/v1/messages")) {
            type = 1;
            protocol = "anthropic";
            overrides.set("gatewayAuthMode", "bearer");
        } else if (paths.contains("/v1beta/models/{model}:generateContent")) {
            type = 1;
            protocol = "gemini";
            overrides.set("gatewayAuthMode", "bearer");
        } else {
            return null;
        }
        String presetCode = type == 1 ? presets.findPresetCode(id, type)
                : presets.findProviderPresetCode("infistar", id, type);
        boolean commonPreset = type != 1 && presetCode == null;
        if (commonPreset) {
            presetCode = type == 2 ? "infistar:common-image" : "infistar:common-video";
        }
        var preset = presetCode == null ? null : presets.getPreset(presetCode);
        if (preset != null && type == 3) {
            overrides = new JSONObject();
        }
        if (preset != null && type == 2) {
            boolean references = paths.contains("/v1/images/edits")
                    && preset.getJSONObject("config").getBool("supportReferenceImages", false);
            overrides.set("supportReferenceImages", references).set("maxReferenceImages", references ? 1 : 0);
        }
        String name = preset == null || commonPreset ? id : preset.getStr("name");
        return new ProviderModelCandidate(id, name, type, protocol, presetCode, overrides.toString(), null);
    }

    private boolean hasMediaGenerationTag(String tags) {
        return tags != null && Arrays.stream(tags.split(","))
                .map(String::trim)
                .anyMatch(Set.of("图片生成", "图像生成", "视频生成")::contains);
    }

    private Set<String> endpointPaths(JSONArray types, JSONObject definitions) {
        var result = new HashSet<String>();
        if (types == null) {
            return result;
        }
        for (String type : types.toList(String.class)) {
            var definition = definitions.getJSONObject(type);
            if (definition != null && "POST".equals(definition.getStr("method"))) {
                result.add(definition.getStr("path"));
            }
        }
        return result;
    }

    private JSONArray requireArray(JSONObject response, String key, String label) {
        JSONArray array = response.getJSONArray(key);
        if (array == null) {
            throw new BusinessException("无限星河" + label + "响应格式无效");
        }
        return array;
    }
}
