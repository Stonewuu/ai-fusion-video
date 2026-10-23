package com.stonewu.fusion.service.generation.transport;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import okhttp3.*;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * 官方 JSON 协议共用传输。POST 只提交一次；只读 GET 可恢复失效的空闲连接。
 */
public final class OfficialVideoHttpClient {
    private static final MediaType JSON = MediaType.get("application/json; charset=utf-8");
    private static final OkHttpClient BASE = GenerationHttpClients.providerRequestBuilder()
            .connectTimeout(60, TimeUnit.SECONDS).readTimeout(120, TimeUnit.SECONDS)
            .writeTimeout(60, TimeUnit.SECONDS).build();
    private static final OkHttpClient READ_BASE = new OkHttpClient.Builder()
            .connectTimeout(60, TimeUnit.SECONDS).readTimeout(120, TimeUnit.SECONDS)
            .writeTimeout(60, TimeUnit.SECONDS).followRedirects(false).followSslRedirects(false)
            .retryOnConnectionFailure(true).build();
    private final OkHttpClient client;
    private final OkHttpClient readClient;
    private final String apiKey;

    public OfficialVideoHttpClient(ApiConfig config) {
        if (StrUtil.isBlank(config.getApiKey())) {
            throw new BusinessException("请配置官方供应商的 API Key");
        }
        client = AiProxySupport.okHttpClient(BASE, config);
        readClient = AiProxySupport.okHttpClient(READ_BASE, config);
        apiKey = config.getApiKey();
    }

    public static String baseUrl(ApiConfig config, String versionPath) {
        if (StrUtil.isBlank(config.getApiUrl())) {
            throw new BusinessException("请填写官方供应商的 API 地址");
        }
        String root = config.getApiUrl().trim().replaceAll("/+$", "");
        HttpUrl parsed;
        try {
            parsed = HttpUrl.get(root);
        } catch (IllegalArgumentException invalid) {
            throw new BusinessException("供应商 API 地址无效");
        }
        if (parsed.query() != null || parsed.fragment() != null) {
            throw new BusinessException("供应商 API 地址不能包含查询参数或页面锚点");
        }
        return root.endsWith(versionPath) ? root : root + versionPath;
    }

    public static String taskUrl(String base, String path, String taskId) {
        if (StrUtil.isBlank(taskId)) {
            throw new BusinessException("缺少供应商任务标识");
        }
        return HttpUrl.get(base + path).newBuilder().addPathSegment(taskId).build().toString();
    }

    public Request post(String url, JSONObject body, Map<String, String> headers) {
        byte[] bytes = body.toString().getBytes(StandardCharsets.UTF_8);
        if (bytes.length > 64 * 1024 * 1024) {
            throw new BusinessException("视频请求超过 64 MB，请减少参考素材或改用公网地址");
        }
        var builder = request(url).post(RequestBody.create(bytes, JSON));
        headers.forEach(builder::header);
        return builder.build();
    }

    public JSONObject get(String url) {
        return execute(request(url).get().build(), readClient);
    }

    public JSONObject execute(Request request) {
        return execute(request, client);
    }

    private JSONObject execute(Request request, OkHttpClient transport) {
        GenerationHttpClients.requireNoTransaction();
        try (Response response = transport.newCall(request).execute()) {
            String body = response.body() == null ? "" : response.body().string();
            if (!response.isSuccessful()) {
                throw GenerationProviderFailure.http(response.code(), body);
            }
            if (!JSONUtil.isTypeJSONObject(body)) {
                throw new BusinessException("供应商返回了无效的任务响应");
            }
            return JSONUtil.parseObj(body);
        } catch (IOException failure) {
            throw new UncheckedIOException("供应商请求连接异常", failure);
        }
    }

    private Request.Builder request(String url) {
        return new Request.Builder().url(url).header("Authorization", "Bearer " + apiKey)
                .header("Accept", "application/json");
    }
}
