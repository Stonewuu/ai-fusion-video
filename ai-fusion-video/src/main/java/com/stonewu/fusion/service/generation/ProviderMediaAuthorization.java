package com.stonewu.fusion.service.generation;

import cn.hutool.core.util.StrUtil;
import com.stonewu.fusion.entity.ai.ApiConfig;

import java.net.URI;

/**
 * 仅向实际提供商同源的生成结果发送密钥。
 */
public final class ProviderMediaAuthorization {
    private ProviderMediaAuthorization() {
    }

    public static boolean shouldAuthorize(String mediaUrl, ApiConfig config, String defaultApiUrl) {
        if (config == null || StrUtil.isBlank(config.getApiKey())) {
            return false;
        }
        String apiUrl = StrUtil.isNotBlank(config.getApiUrl()) ? config.getApiUrl() : defaultApiUrl;
        if (StrUtil.isBlank(apiUrl)) {
            return false;
        }
        try {
            URI media = URI.create(mediaUrl);
            URI api = URI.create(apiUrl.trim());
            return ("https".equalsIgnoreCase(media.getScheme()) || "http".equalsIgnoreCase(media.getScheme()))
                    && media.getScheme().equalsIgnoreCase(api.getScheme())
                    && media.getHost() != null && api.getHost() != null
                    && media.getHost().equalsIgnoreCase(api.getHost())
                    && effectivePort(media) == effectivePort(api);
        } catch (IllegalArgumentException e) {
            return false;
        }
    }

    private static int effectivePort(URI uri) {
        if (uri.getPort() >= 0) {
            return uri.getPort();
        }
        return "https".equalsIgnoreCase(uri.getScheme()) ? 443 : 80;
    }
}
