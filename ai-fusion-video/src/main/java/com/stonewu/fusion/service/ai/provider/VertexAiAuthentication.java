package com.stonewu.fusion.service.ai.provider;

import cn.hutool.core.util.StrUtil;
import com.google.auth.oauth2.GoogleCredentials;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

/** Vertex 两种鉴权明确分开，不能因为另一种凭据存在而自动切换。 */
public final class VertexAiAuthentication {
    public static final String API_KEY_PLATFORM = "vertex_ai_api_key";
    private static final String SCOPE = "https://www.googleapis.com/auth/cloud-platform";

    private VertexAiAuthentication() {
    }

    public static boolean usesApiKey(ApiConfig config) {
        return config != null && API_KEY_PLATFORM.equals(config.getPlatform());
    }

    public static String requireApiKey(ApiConfig config) {
        if (StrUtil.isBlank(config.getApiKey())) {
            throw new BusinessException("请填写 Vertex AI API Key");
        }
        return config.getApiKey().trim();
    }

    public static GoogleCredentials credentials(ApiConfig config) {
        String json = config == null ? null : config.getAppSecret();
        var transport = AiProxySupport.googleHttpTransportFactory(config);
        try {
            GoogleCredentials credentials;
            if (StrUtil.isNotBlank(json)) {
                var input = new ByteArrayInputStream(json.getBytes(StandardCharsets.UTF_8));
                credentials = transport == null ? GoogleCredentials.fromStream(input)
                        : GoogleCredentials.fromStream(input, transport);
            } else {
                credentials = transport == null ? GoogleCredentials.getApplicationDefault()
                        : GoogleCredentials.getApplicationDefault(transport);
            }
            return credentials.createScoped(List.of(SCOPE));
        } catch (IOException failure) {
            throw new BusinessException("Vertex AI 凭据无法加载，请检查服务账号 JSON 或服务器的默认凭据");
        }
    }

    public static Map<String, String> headers(ApiConfig config) {
        if (usesApiKey(config)) {
            return Map.of("x-goog-api-key", requireApiKey(config));
        }
        var credentials = credentials(config);
        try {
            credentials.refreshIfExpired();
            return Map.of("Authorization", "Bearer " + credentials.getAccessToken().getTokenValue());
        } catch (IOException failure) {
            throw new BusinessException("Vertex AI 获取访问令牌失败，请检查服务账号权限及 Google OAuth 连接");
        }
    }

    public static String modelUrl(ApiConfig config, String model, String operation) {
        if (!model.matches("[A-Za-z0-9._@-]+")) {
            throw new BusinessException("请填写 Vertex 模型 ID，不要填写完整资源地址");
        }
        if (usesApiKey(config)) {
            return "https://aiplatform.googleapis.com/v1/publishers/google/models/" + model + ":" + operation;
        }
        String project = config.getAppId();
        String location = StrUtil.blankToDefault(config.getApiUrl(), "us-central1");
        if (StrUtil.isBlank(project) || !project.matches("[A-Za-z0-9._:-]+") || !location.matches("[a-z0-9-]+")) {
            throw new BusinessException("请检查 Vertex 项目 ID 和区域配置");
        }
        String host = switch (location) {
            case "global" -> "aiplatform.googleapis.com";
            case "us", "eu" -> "aiplatform." + location + ".rep.googleapis.com";
            default -> location + "-aiplatform.googleapis.com";
        };
        return "https://" + host + "/v1/projects/" + project + "/locations/" + location
                + "/publishers/google/models/" + model + ":" + operation;
    }
}
