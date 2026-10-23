package com.stonewu.fusion.service.ai.catalog;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.ApiConfig;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

/** 服务端统一维护提供商身份、赞助资料和快捷配置默认值。 */
@Service
public class ProviderCatalogService {
    private final Map<String, ProviderDefinition> providers;

    public ProviderCatalogService(ObjectMapper objectMapper) throws IOException {
        try (var input = new ClassPathResource("provider-catalog/providers.json").getInputStream()) {
            List<ProviderDefinition> definitions = objectMapper.readValue(input, new TypeReference<>() {});
            var indexed = new LinkedHashMap<String, ProviderDefinition>();
            for (ProviderDefinition definition : definitions) {
                if (definition.id() == null || definition.id().isBlank()
                        || indexed.putIfAbsent(definition.id(), definition) != null) {
                    throw new IllegalStateException("提供商目录包含空值或重复标识");
                }
                if (definition.sponsored() && (definition.credentialUrl() == null
                        || !definition.credentialUrl().startsWith("https://"))) {
                    throw new IllegalStateException("赞助提供商必须配置获取密钥链接: " + definition.id());
                }
                if (definition.supportedModelTypes() == null || definition.modelPresetCodes() == null) {
                    throw new IllegalStateException("提供商目录缺少能力或模型定义: " + definition.id());
                }
            }
            providers = Map.copyOf(indexed);
        }
    }

    @Cacheable(value = "providerCatalog", key = "'catalog-v1'")
    public List<ProviderDefinition> list() {
        return providers.values().stream()
                .sorted(Comparator.comparing(ProviderDefinition::sponsored).reversed()
                        .thenComparing(ProviderDefinition::id))
                .toList();
    }

    public ProviderDefinition require(String id) {
        if (id == null || id.isBlank()) {
            throw new BusinessException(400, "请选择提供商");
        }
        ProviderDefinition definition = providers.get(id);
        if (definition == null) {
            throw new BusinessException(400, "提供商不存在，请重新选择");
        }
        return definition;
    }

    public void validateBinding(ApiConfig config) {
        if (config.getProviderId() == null) {
            return;
        }
        if (!require(config.getProviderId()).platform().equals(config.getPlatform())) {
            throw new BusinessException(400, "提供商与接入类型不匹配，请重新选择提供商或改为自定义接入");
        }
        if (require(config.getProviderId()).quickSetup()
                && !Integer.valueOf(0).equals(config.getStatus())
                && (config.getApiKey() == null || config.getApiKey().isBlank())) {
            throw new BusinessException(400, "请填写平台的 API Key");
        }
    }
}
