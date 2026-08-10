package com.stonewu.fusion.service.storage;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONObject;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.storage.StorageConfig;
import org.springframework.stereotype.Component;

/**
 * Resolves persisted storage configuration into concrete S3 SDK settings.
 */
@Component
public class S3StorageConfigResolver {

    private final StorageProviderRegistry providerRegistry;

    public S3StorageConfigResolver(StorageProviderRegistry providerRegistry) {
        this.providerRegistry = providerRegistry;
    }

    public ResolvedS3StorageConfig resolve(StorageConfig config) {
        if (config == null) {
            throw new BusinessException("S3 存储配置为空");
        }
        if (!StorageTypes.isS3Like(config.getType())) {
            throw new BusinessException("不是 S3 兼容存储配置: " + config.getType());
        }

        String provider = resolveProvider(config);
        StorageProviderProfile profile = providerRegistry.get(provider);
        JSONObject options = StorageConfigOptions.parseOrEmpty(config.getOptions());

        String region = StrUtil.blankToDefault(config.getRegion(), profile.defaultRegion());
        String endpoint = resolveEndpoint(config.getEndpoint(), profile, region);
        String signingRegion = StrUtil.blankToDefault(
                StorageConfigOptions.getString(options, "signingRegion"),
                StrUtil.blankToDefault(region, "us-east-1")
        );

        validateRequired(config, endpoint, profile);

        return new ResolvedS3StorageConfig(
                config,
                profile,
                provider,
                endpoint,
                config.getBucketName(),
                config.getAccessKey(),
                config.getSecretKey(),
                region,
                signingRegion,
                config.getBasePath(),
                config.getCustomDomain(),
                StorageConfigOptions.getBoolean(options, "pathStyleAccessEnabled", profile.pathStyleAccessEnabled()),
                StorageConfigOptions.getBoolean(options, "chunkedEncodingEnabled", profile.chunkedEncodingEnabled()),
                StrUtil.blankToDefault(StorageConfigOptions.getString(options, "checksumCalculation"), "WHEN_REQUIRED")
        );
    }

    public String resolveProvider(StorageConfig config) {
        String legacyProvider = StorageTypes.legacyProvider(config.getType());
        if (StrUtil.isNotBlank(legacyProvider)) {
            return legacyProvider;
        }
        return StrUtil.blankToDefault(config.getProvider(), StorageProviderRegistry.GENERIC_S3);
    }

    private String resolveEndpoint(String configuredEndpoint, StorageProviderProfile profile, String region) {
        if (StrUtil.isNotBlank(configuredEndpoint)) {
            return normalizeEndpoint(configuredEndpoint);
        }
        if (StrUtil.isBlank(profile.endpointTemplate())) {
            return null;
        }
        if (StrUtil.isBlank(region)) {
            throw new BusinessException(profile.label() + " 未配置 endpoint 时必须填写 region");
        }
        return normalizeEndpoint(profile.endpointTemplate().replace("{region}", region));
    }

    public String normalizeEndpoint(String endpoint) {
        if (StrUtil.isBlank(endpoint)) {
            return null;
        }
        String value = endpoint.trim();
        if (!value.startsWith("http://") && !value.startsWith("https://")) {
            value = "https://" + value;
        }
        return value.replaceAll("/+$", "");
    }

    private void validateRequired(StorageConfig config, String endpoint, StorageProviderProfile profile) {
        if (StrUtil.isBlank(endpoint) && profile.endpointRequired()) {
            throw new BusinessException("S3 存储配置缺少 endpoint");
        }
        if (StrUtil.isBlank(config.getBucketName())) {
            throw new BusinessException("S3 存储配置缺少 bucketName");
        }
        if (StrUtil.isBlank(config.getAccessKey()) || StrUtil.isBlank(config.getSecretKey())) {
            throw new BusinessException("S3 存储配置缺少 accessKey 或 secretKey");
        }
    }
}
