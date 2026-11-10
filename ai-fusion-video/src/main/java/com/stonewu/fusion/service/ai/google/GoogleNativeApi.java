package com.stonewu.fusion.service.ai.google;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.provider.VertexAiAuthentication;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import com.stonewu.fusion.service.generation.transport.GenerationHttpClients;
import com.stonewu.fusion.service.generation.transport.GenerationProviderFailure;
import okhttp3.HttpUrl;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import org.springframework.stereotype.Component;
import lombok.extern.slf4j.Slf4j;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/** Google 原生接口共用鉴权和单次传输，异步任务由平台调度查询。 */
@Component
@Slf4j
public class GoogleNativeApi {
    private static final OkHttpClient CLIENT = GenerationHttpClients.singleRequestBuilder()
            .connectTimeout(30, TimeUnit.SECONDS).readTimeout(5, TimeUnit.MINUTES)
            .writeTimeout(2, TimeUnit.MINUTES).build();

    public boolean developerApi(ApiConfig api) {
        return "gemini".equals(api.getPlatform());
    }

    public Map<String, String> headers(ApiConfig api) {
        if (developerApi(api)) {
            if (StrUtil.isBlank(api.getApiKey())) {
                throw new BusinessException("请填写 Gemini API Key");
            }
            return Map.of("x-goog-api-key", api.getApiKey());
        }
        return VertexAiAuthentication.headers(api);
    }

    public String developerBase(ApiConfig api) {
        return StrUtil.blankToDefault(api.getApiUrl(), "https://generativelanguage.googleapis.com")
                .replaceAll("/+$", "").replaceAll("/v1(?:beta)?$", "") + "/v1beta";
    }

    public String modelUrl(ApiConfig api, String model, String operation) {
        requireIdentifier(model);
        return developerApi(api) ? developerBase(api) + "/models/" + model + ":" + operation
                : VertexAiAuthentication.modelUrl(api, model, operation);
    }

    public String interactionsUrl(ApiConfig api) {
        if (developerApi(api)) {
            return developerBase(api) + "/interactions";
        }
        String project = api.getAppId();
        String location = StrUtil.blankToDefault(api.getApiUrl(), "global");
        if (StrUtil.isBlank(project) || !project.matches("[A-Za-z0-9._:-]+")) {
            throw new BusinessException("使用此模型前，请在 Vertex API 配置中填写项目 ID");
        }
        String host = switch (location) {
            case "global" -> "aiplatform.googleapis.com";
            case "us", "eu" -> "aiplatform." + location + ".rep.googleapis.com";
            default -> throw new BusinessException("此模型支持 global、us 或 eu 区域，请调整 Vertex 区域");
        };
        return "https://" + host + "/v1/projects/" + project + "/locations/" + location + "/interactions";
    }

    public String interactionUrl(ApiConfig api, String id) {
        return HttpUrl.get(interactionsUrl(api)).newBuilder().addPathSegment(id).build().toString();
    }

    public Request post(ApiConfig api, String url, JSONObject body) {
        byte[] encoded = body.toString().getBytes(StandardCharsets.UTF_8);
        if (encoded.length > 64L * 1024 * 1024) {
            throw new BusinessException("输入内容超过 64 MB，请减少参考素材");
        }
        return request(api, url).post(RequestBody.create(encoded, MediaType.get("application/json"))).build();
    }

    public JSONObject postJson(ApiConfig api, String url, JSONObject body) {
        return execute(api, post(api, url, body));
    }

    public JSONObject getJson(ApiConfig api, String url) {
        return execute(api, request(api, url).get().build());
    }

    public JSONObject execute(ApiConfig api, Request request) {
        GenerationHttpClients.requireNoTransaction();
        try (var response = AiProxySupport.okHttpClient(CLIENT, api).newCall(request).execute()) {
            String body = response.body() == null ? "" : response.body().string();
            if (!response.isSuccessful()) {
                var failure = GenerationProviderFailure.http(response.code(), body);
                String diagnostic = failure.reason().replaceAll("AIza[A-Za-z0-9_-]+|AQ\\.[A-Za-z0-9_.-]+", "[凭据已隐藏]");
                log.warn("Google 请求失败: status={}, reason={}", response.code(), diagnostic);
                throw GenerationProviderFailure.http(response.code(), GoogleRequestErrors.message(response.code(), failure.reason()));
            }
            if (!JSONUtil.isTypeJSONObject(body)) {
                throw new BusinessException("Google 返回的结果无法读取，请重试");
            }
            return JSONUtil.parseObj(body);
        } catch (IOException failure) {
            throw new GoogleRequestUncertainException(failure);
        }
    }

    private Request.Builder request(ApiConfig api, String url) {
        var request = new Request.Builder().url(url).header("Accept", "application/json");
        headers(api).forEach(request::header);
        return request;
    }

    private void requireIdentifier(String model) {
        if (model == null || !model.matches("[A-Za-z0-9._@-]+")) {
            throw new BusinessException("请填写有效的 Google 模型 ID");
        }
    }
}
