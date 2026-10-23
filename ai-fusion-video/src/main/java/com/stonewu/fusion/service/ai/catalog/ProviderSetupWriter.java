package com.stonewu.fusion.service.ai.catalog;

import cn.hutool.crypto.digest.DigestUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.mapper.ai.AiModelMapper;
import com.stonewu.fusion.mapper.ai.ApiConfigMapper;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.ModelPresetService;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

/** 只在候选模型发现完成后执行短事务；没有外部网络请求。 */
@Service
@RequiredArgsConstructor
public class ProviderSetupWriter {
    private final ApiConfigService configs;
    private final ApiConfigMapper configMapper;
    private final AiModelService models;
    private final AiModelMapper modelMapper;
    private final ModelPresetService presets;

    @Transactional
    @CacheEvict(value = "episodeBgmConfig", allEntries = true)
    public ProviderSetupResult save(ApiConfig config, List<ProviderModelCandidate> selected,
                                     String requestKey, String requestHash) {
        if (config.getId() == null) {
            config.setProviderSetupKey(requestKey);
            config.setProviderSetupHash(requestHash);
            configs.createApiConfig(config);
        }
        var locked = configMapper.selectByIdForUpdate(config.getId());
        if (locked == null) {
            throw new BusinessException(404, "API 配置不存在");
        }
        if (!Objects.equals(connectionFingerprint(locked), connectionFingerprint(config))) {
            throw new BusinessException(409, "连接配置已经变化，请重新读取模型列表");
        }
        var existing = modelMapper.selectList(new LambdaQueryWrapper<AiModel>()
                        .eq(AiModel::getApiConfigId, config.getId())).stream().map(AiModel::getCode)
                .collect(Collectors.toCollection(HashSet::new));
        int added = 0;
        for (var candidate : selected) {
            if (existing.add(candidate.id())) {
                models.createAiModel(toModel(config.getId(), candidate));
                added++;
            }
        }
        return new ProviderSetupResult(config.getId(), added, selected.size() - added);
    }

    private AiModel toModel(Long configId, ProviderModelCandidate candidate) {
        var builder = AiModel.builder().apiConfigId(configId).name(candidate.name()).code(candidate.id())
                .modelType(candidate.modelType()).modelProtocol(candidate.modelProtocol())
                .capabilityPresetCode(candidate.capabilityPresetCode()).config(candidate.config());
        if (candidate.capabilityPresetCode() != null) {
            var preset = presets.getPreset(candidate.capabilityPresetCode());
            builder.description(preset.getStr("description"));
            if (candidate.modelType() == 1) {
                List<String> inputs = preset.getJSONArray("multimodalInputTypes").toList(String.class);
                builder.supportVision(inputs.contains("image")).multimodalInputTypes(inputs)
                        .multimodalInputTransports(inputs.stream().collect(Collectors.toMap(type -> type,
                                type -> preset.getJSONObject("multimodalInputTransports").getJSONArray(type)
                                        .toList(String.class))))
                        .contextWindow(preset.getInt("contextWindow"))
                        .supportReasoning(preset.getBool("supportReasoning", false))
                        .reasoningEffortLevels(preset.getJSONArray("reasoningEffortLevels").toList(String.class));
            }
        }
        return builder.build();
    }

    private String connectionFingerprint(ApiConfig config) {
        return DigestUtil.sha256Hex(String.join("\u0000", String.valueOf(config.getProviderId()),
                String.valueOf(config.getPlatform()), String.valueOf(config.getApiUrl()),
                String.valueOf(config.getApiKey()), String.valueOf(config.getProxyType()),
                String.valueOf(config.getProxyHost()), String.valueOf(config.getProxyPort()),
                String.valueOf(config.getProxyUsername()), String.valueOf(config.getProxyPassword()),
                String.valueOf(config.getStatus())));
    }
}
