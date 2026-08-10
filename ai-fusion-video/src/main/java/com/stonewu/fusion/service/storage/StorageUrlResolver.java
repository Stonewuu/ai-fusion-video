package com.stonewu.fusion.service.storage;

import cn.hutool.core.util.StrUtil;
import com.stonewu.fusion.common.BusinessException;
import org.springframework.stereotype.Component;

import java.net.URI;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

/**
 * Builds public object URLs for uploaded media.
 */
@Component
public class StorageUrlResolver {

    public String resolvePublicUrl(ResolvedS3StorageConfig config, String objectKey) {
        String encodedKey = encodeObjectKey(objectKey);

        if (StrUtil.isNotBlank(config.customDomain())) {
            String domain = ensureScheme(config.customDomain()).replaceAll("/+$", "");
            return domain + "/" + encodedKey;
        }

        StorageProviderProfile profile = config.profile();
        if (StrUtil.isNotBlank(profile.publicUrlTemplate()) && StrUtil.isNotBlank(config.region())) {
            return profile.publicUrlTemplate()
                    .replace("{bucket}", config.bucketName())
                    .replace("{region}", config.region())
                    .replace("{key}", encodedKey);
        }

        if (!profile.publicUrlInferable()) {
            throw new BusinessException(profile.label() + " 无法可靠推导公开访问地址，请配置自定义域名/CDN 域名");
        }

        if (StrUtil.isBlank(config.endpoint())) {
            throw new BusinessException("S3 存储配置缺少 endpoint，无法生成公开访问地址");
        }

        URI endpoint = URI.create(config.endpoint());
        String base = endpoint.getScheme() + "://" + endpoint.getHost();
        if (endpoint.getPort() > 0) {
            base += ":" + endpoint.getPort();
        }

        if (config.pathStyleAccessEnabled()) {
            return base + "/" + config.bucketName() + "/" + encodedKey;
        }
        return endpoint.getScheme() + "://" + config.bucketName() + "." + endpoint.getAuthority() + "/" + encodedKey;
    }

    private String ensureScheme(String value) {
        String trimmed = value.trim();
        if (!trimmed.startsWith("http://") && !trimmed.startsWith("https://")) {
            return "https://" + trimmed;
        }
        return trimmed;
    }

    private String encodeObjectKey(String objectKey) {
        String normalized = objectKey.replace("\\", "/").replaceAll("^/+", "");
        if (normalized.isBlank()) {
            return "";
        }
        return String.join("/", Arrays.stream(normalized.split("/"))
                .map(part -> URLEncoder.encode(part, StandardCharsets.UTF_8).replace("+", "%20"))
                .toList());
    }
}
