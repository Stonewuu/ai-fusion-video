package com.stonewu.fusion.service.ai.apixo;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.generation.transport.OfficialVideoHttpClient;
import okhttp3.HttpUrl;

/** APIXO 对话网关与生成服务使用独立地址；自定义网关沿用用户配置的根地址。 */
public final class ApixoEndpoints {
    private ApixoEndpoints() {
    }

    public static String generationBase(ApiConfig config) {
        return OfficialVideoHttpClient.baseUrl(config, "/api/v1");
    }

    public static String chatBase(ApiConfig config) {
        HttpUrl configured = HttpUrl.get(generationBase(config));
        if ("api.apixo.ai".equalsIgnoreCase(configured.host())) {
            if (!configured.isHttps() || configured.port() != 443 || !configured.username().isEmpty()
                    || !configured.password().isEmpty() || !"/api/v1".equals(configured.encodedPath())) {
                throw new BusinessException("APIXO 官方 API 地址应为 https://api.apixo.ai");
            }
            return "https://llm.apixo.ai/v1";
        }
        String root = generationBase(config);
        return root.substring(0, root.length() - "/api/v1".length()) + "/v1";
    }

    public static String submit(ApiConfig config, String model) {
        return HttpUrl.get(generationBase(config) + "/generateTask").newBuilder()
                .addPathSegment(model).build().toString();
    }

    public static String status(ApiConfig config, String model, String taskId) {
        return HttpUrl.get(generationBase(config) + "/statusTask").newBuilder()
                .addPathSegment(model).addQueryParameter("taskId", taskId).build().toString();
    }
}
