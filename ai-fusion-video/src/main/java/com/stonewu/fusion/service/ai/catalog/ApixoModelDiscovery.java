package com.stonewu.fusion.service.ai.catalog;

import cn.hutool.json.JSONObject;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.ModelPresetService;
import com.stonewu.fusion.service.ai.apixo.ApixoEndpoints;
import com.stonewu.fusion.service.ai.apixo.ApixoTasks;
import com.stonewu.fusion.service.generation.transport.OfficialVideoHttpClient;
import com.stonewu.fusion.service.generation.transport.GenerationProviderFailure;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

/** 只读接口验证密钥；模型目录采用已核对的 APIXO 公开模型及参数预设。 */
@Component
@RequiredArgsConstructor
public class ApixoModelDiscovery implements ProviderModelDiscovery {
    private final ModelPresetService presets;

    @Override
    public String type() {
        return "apixo";
    }

    @Override
    public ProviderDiscoveryResult discover(ApiConfig config, ProviderDefinition provider) {
        JSONObject response;
        try {
            response = new OfficialVideoHttpClient(config).get(
                    ApixoEndpoints.generationBase(config) + "/apikeys/current-balance");
            ApixoTasks.requireSuccess(response);
        } catch (GenerationProviderFailure failure) {
            String message = switch (failure.httpStatus()) {
                case 401 -> "APIXO 密钥无效，请检查后重试";
                case 403 -> "APIXO 密钥没有访问权限，请检查平台中的密钥设置";
                case 429 -> "APIXO 请求过于频繁，请稍后重试";
                default -> "APIXO 模型目录读取失败，请检查 API 地址或稍后重试";
            };
            throw new BusinessException(502, message);
        }
        if (!(response.get("data") instanceof Number)) {
            throw new BusinessException("APIXO 密钥校验未返回有效余额，请检查 API 地址与密钥");
        }
        var models = provider.modelPresetCodes().stream().map(code -> {
            JSONObject preset = presets.getPreset(code);
            if (preset == null || !provider.id().equals(preset.getStr("providerId"))) {
                throw new IllegalStateException("APIXO 模型目录引用了无效预设: " + code);
            }
            return new ProviderModelCandidate(preset.getStr("modelCode"), preset.getStr("name"),
                    preset.getInt("modelType"), preset.getStr("modelProtocol"), code,
                    preset.getJSONObject("config").toString(), null, null);
        }).toList();
        return new ProviderDiscoveryResult(provider.id(), true, "选择需要的对话、图片或视频模型。", models);
    }
}
