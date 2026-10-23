package com.stonewu.fusion.service.ai.catalog;

import cn.hutool.core.util.StrUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.controller.ai.vo.ApiConfigSaveReqVO;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.generation.transport.GenerationProviderFailure;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.io.UncheckedIOException;

/** 外部模型发现不持有事务；候选列表不缓存密钥或写入业务配置。 */
@Service
@RequiredArgsConstructor
public class ProviderDiscoveryService {
    private final ProviderCatalogService catalog;
    private final ApiConfigService configs;
    private final List<ProviderModelDiscovery> discoveries;

    @Transactional(propagation = Propagation.NEVER)
    public ProviderDiscoveryResult preview(ApiConfigSaveReqVO request) {
        ApiConfig config = resolveConfiguration(request);
        ProviderDefinition provider = catalog.require(config.getProviderId());
        ProviderModelDiscovery discovery = discoveries.stream()
                .filter(candidate -> candidate.type().equals(provider.discovery())).findFirst()
                .orElseThrow(() -> new BusinessException("此提供商请使用常规配置或工作流管理添加模型"));
        try {
            return discovery.discover(config, provider);
        } catch (GenerationProviderFailure failure) {
            throw new BusinessException(502, "读取模型失败（HTTP " + failure.httpStatus() + "）：" + failure.reason());
        } catch (UncheckedIOException failure) {
            throw new BusinessException(502, "读取模型失败，请检查 API 地址与网络连接后重试");
        }
    }

    public ApiConfig resolveConfiguration(ApiConfigSaveReqVO request) {
        ProviderDefinition provider = catalog.require(request.getProviderId());
        ApiConfig saved = null;
        if (request.getId() != null) {
            saved = configs.getById(request.getId());
            if (saved == null) {
                throw new BusinessException(404, "API 配置不存在");
            }
            if (!provider.id().equals(saved.getProviderId())) {
                throw new BusinessException(400, "追加模型时不能切换已有配置的提供商");
            }
            if (!Integer.valueOf(1).equals(saved.getStatus())) {
                throw new BusinessException(400, "请先启用此 API 配置，再添加模型");
            }
            if ((request.getApiKey() != null && !request.getApiKey().trim().equals(saved.getApiKey()))
                    || (request.getApiUrl() != null && !request.getApiUrl().trim().equals(saved.getApiUrl()))) {
                throw new BusinessException(400, "请先在编辑 API 配置中保存连接变更，再添加模型");
            }
        }
        String key = request.getApiKey() == null && saved != null ? saved.getApiKey() : request.getApiKey();
        if (StrUtil.isBlank(key)) {
            throw new BusinessException(400, "请填写 API Key");
        }
        var config = ApiConfig.builder().id(request.getId()).providerId(provider.id()).platform(provider.platform())
                .name(request.getName()).apiKey(key.trim())
                .apiUrl(request.getApiUrl() == null ? (saved == null ? provider.url() : saved.getApiUrl())
                        : request.getApiUrl().trim())
                .textProtocol(provider.textProtocol()).imageProtocol(provider.imageProtocol())
                .videoProtocol(provider.videoProtocol()).autoAppendV1Path(provider.appendV1())
                .status(1).build();
        if (saved != null) {
            config.setProxyType(saved.getProxyType());
            config.setProxyHost(saved.getProxyHost());
            config.setProxyPort(saved.getProxyPort());
            config.setProxyUsername(saved.getProxyUsername());
            config.setProxyPassword(saved.getProxyPassword());
        }
        return config;
    }
}
