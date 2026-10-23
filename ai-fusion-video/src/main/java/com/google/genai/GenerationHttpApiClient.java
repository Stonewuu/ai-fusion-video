package com.google.genai;

import com.google.auth.oauth2.GoogleCredentials;
import com.google.genai.errors.GenAiIOException;
import com.google.genai.types.HttpOptions;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.provider.SingleRequestModelSupport;
import okhttp3.*;

import java.io.IOException;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;

/**
 * Google GenAI 1.45 的模型序列化与鉴权仍由 SDK 完成，只替换不可注入的 HTTP 发送点。
 * SDK 的响应类型为包内可见，因此此适配器与它同包；升级 SDK 必须通过单次交换测试。
 */
public final class GenerationHttpApiClient extends HttpApiClient {
    private final OkHttpClient single;

    public GenerationHttpApiClient(String key, HttpOptions options, ApiConfig api) {
        super(Optional.of(key), Optional.of(options), Optional.empty());
        single = SingleRequestModelSupport.client(api);
    }

    public GenerationHttpApiClient(String project, String location, GoogleCredentials credentials,
                                   HttpOptions options, ApiConfig api) {
        super(Optional.empty(), Optional.of(project), Optional.of(location), Optional.ofNullable(credentials),
                Optional.of(options), Optional.empty());
        single = SingleRequestModelSupport.client(api);
    }

    @Override
    public HttpApiResponse request(String method, String path, String body, Optional<HttpOptions> options) {
        return execute(buildRequest(method, path, body, options));
    }

    @Override
    public HttpApiResponse request(String method, String path, byte[] body, Optional<HttpOptions> options) {
        return execute(buildRequest(method, path, body, options));
    }

    private HttpApiResponse execute(Request request) {
        try {
            return new HttpApiResponse(single.newCall(request).execute());
        } catch (IOException error) {
            throw new GenAiIOException("文本模型的单次请求未完成", error);
        }
    }

    @Override
    public CompletableFuture<ApiResponse> asyncRequest(String method, String path, String body,
                                                       Optional<HttpOptions> options) {
        return executeAsync(buildRequest(method, path, body, options));
    }

    @Override
    public CompletableFuture<ApiResponse> asyncRequest(String method, String path, byte[] body,
                                                       Optional<HttpOptions> options) {
        return executeAsync(buildRequest(method, path, body, options));
    }

    private CompletableFuture<ApiResponse> executeAsync(Request request) {
        var result = new CompletableFuture<ApiResponse>();
        Call call = single.newCall(request);
        call.enqueue(new Callback() {
            @Override
            public void onFailure(Call ignored, IOException error) {
                result.completeExceptionally(error);
            }

            @Override
            public void onResponse(Call ignored, Response response) {
                if (!result.complete(new HttpApiResponse(response))) {
                    response.close();
                }
            }
        });
        result.whenComplete((value, error) -> {
            if (result.isCancelled()) {
                call.cancel();
            }
        });
        return result;
    }

    @Override
    public void close() {
        single.dispatcher().cancelAll();
        super.close();
        single.dispatcher().executorService().shutdown();
        single.connectionPool().evictAll();
    }
}
