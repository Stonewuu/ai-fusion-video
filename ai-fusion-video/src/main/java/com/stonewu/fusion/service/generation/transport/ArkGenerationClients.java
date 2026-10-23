package com.stonewu.fusion.service.generation.transport;

import cn.hutool.core.util.StrUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import com.volcengine.ark.runtime.interceptor.AuthenticationInterceptor;
import com.volcengine.ark.runtime.interceptor.BatchInterceptor;
import com.volcengine.ark.runtime.interceptor.EncryptionInterceptor;
import com.volcengine.ark.runtime.interceptor.RequestIdInterceptor;
import com.volcengine.ark.runtime.service.ArkApi;
import com.volcengine.ark.runtime.service.ArkService;
import okhttp3.ConnectionPool;

import java.util.concurrent.TimeUnit;

/**
 * 保留 Ark 的认证与序列化协议，不安装其自动重试拦截器。
 */
public final class ArkGenerationClients {
    private ArkGenerationClients() {
    }

    public static ArkService create(ApiConfig config) {
        if (StrUtil.isBlank(config.getApiKey())) {
            throw new BusinessException("火山引擎缺少 API Key 配置");
        }
        String baseUrl = StrUtil.isBlank(config.getApiUrl()) ? ArkService.BASE_URL : config.getApiUrl().trim();
        if (!baseUrl.endsWith("/")) {
            baseUrl += "/";
        }
        var builder = GenerationHttpClients.providerRequestBuilder()
                .addInterceptor(new AuthenticationInterceptor(config.getApiKey()))
                .addInterceptor(new EncryptionInterceptor(config.getApiKey(), baseUrl))
                .addInterceptor(new RequestIdInterceptor())
                .addInterceptor(new BatchInterceptor())
                .connectionPool(new ConnectionPool(5, 1, TimeUnit.SECONDS))
                .connectTimeout(1, TimeUnit.MINUTES)
                .readTimeout(25, TimeUnit.MINUTES)
                .callTimeout(25, TimeUnit.MINUTES)
                .writeTimeout(1, TimeUnit.MINUTES);
        var client = AiProxySupport.applyToOkHttp(builder, config).build();
        var retrofit = ArkService.defaultRetrofit(client, ArkService.defaultObjectMapper(), baseUrl, null);
        return new ArkService(retrofit.create(ArkApi.class), client.dispatcher().executorService());
    }
}
