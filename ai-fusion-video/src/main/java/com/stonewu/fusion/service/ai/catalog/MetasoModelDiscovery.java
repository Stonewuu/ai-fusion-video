package com.stonewu.fusion.service.ai.catalog;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.ModelPresetService;
import com.stonewu.fusion.service.generation.transport.OfficialVideoHttpClient;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

/** 使用只读余额接口校验密钥；模型来自已核实的公开 API 模型清单。 */
@Component
@RequiredArgsConstructor
public class MetasoModelDiscovery implements ProviderModelDiscovery {
    private final ModelPresetService presets;

    @Override
    public String type() {
        return "metaso";
    }

    @Override
    public ProviderDiscoveryResult discover(ApiConfig config, ProviderDefinition provider) {
        var result = new OfficialVideoHttpClient(config).get(
                OfficialVideoHttpClient.baseUrl(config, "/v2") + "/credits/balance");
        if (!(result.get("balance") instanceof Number)) {
            throw new BusinessException("秘塔密钥校验未返回有效余额，请检查 API 地址和密钥");
        }
        var models = provider.modelPresetCodes().stream().map(code -> {
            var preset = presets.getPreset(code);
            if (preset == null) {
                throw new IllegalStateException("提供商引用的模型预设不存在: " + code);
            }
            return new ProviderModelCandidate(preset.getStr("modelCode"), preset.getStr("name"),
                    preset.getInt("modelType"), preset.getStr("modelProtocol"), code, "{}", null);
        }).toList();
        return new ProviderDiscoveryResult(provider.id(), true, "密钥已验证；以下为平台公开的视频模型预设。", models);
    }
}
