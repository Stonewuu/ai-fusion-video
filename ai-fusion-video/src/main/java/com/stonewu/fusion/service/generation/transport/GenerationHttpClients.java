package com.stonewu.fusion.service.generation.transport;

import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.RequestBody;
import okio.BufferedSink;
import org.springframework.web.client.RestClient;
import org.springframework.transaction.IllegalTransactionStateException;
import org.springframework.transaction.support.TransactionSynchronizationManager;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.zip.GZIPInputStream;

/**
 * 生成请求的一次传输边界；任何重查或重提均由持久化执行器决定。
 */
public final class GenerationHttpClients {
    private static final OkHttpClient REST_BASE = providerRequestBuilder()
            .connectTimeout(1, TimeUnit.MINUTES)
            .readTimeout(25, TimeUnit.MINUTES)
            .writeTimeout(1, TimeUnit.MINUTES)
            .build();

    private GenerationHttpClients() {
    }

    public static void requireNoTransaction() {
        if (TransactionSynchronizationManager.isActualTransactionActive()) {
            throw new IllegalTransactionStateException("供应商网络请求不能持有数据库事务");
        }
    }

    public static OkHttpClient.Builder singleRequestBuilder() {
        return new OkHttpClient.Builder()
                .retryOnConnectionFailure(false)
                .followRedirects(false)
                .followSslRedirects(false)
                .addInterceptor(chain -> {
                    var request = chain.request();
                    var builder = request.newBuilder().tag(ExchangePermit.class, new ExchangePermit());
                    if (request.body() != null) {
                        builder.method(request.method(), new SingleUseBody(request.body()));
                    }
                    return chain.proceed(builder.build());
                })
                .addNetworkInterceptor(chain -> {
                    var permit = chain.request().tag(ExchangePermit.class);
                    if (permit == null || !permit.used.compareAndSet(false, true)) {
                        throw new IOException("生成传输禁止自动重复发送，请由任务状态重新调度");
                    }
                    return chain.proceed(chain.request());
                });
    }

    public static RestClient.Builder restClientBuilder(ApiConfig apiConfig) {
        var client = AiProxySupport.okHttpClient(REST_BASE, apiConfig);
        return RestClient.builder().requestFactory(new GenerationHttpRequestFactory(client));
    }

    /**
     * 生成服务明确返回的非成功响应有独立类型，SDK 包装后仍可识别其失败事实。
     */
    public static OkHttpClient.Builder providerRequestBuilder() {
        return singleRequestBuilder().addNetworkInterceptor(chain -> {
            var response = chain.proceed(chain.request());
            if (response.isSuccessful() || !chain.request().method().equals("POST")) {
                return response;
            }
            try (response) {
                String body;
                try {
                    byte[] bytes = response.peekBody(16 * 1024).bytes();
                    if ("gzip".equalsIgnoreCase(response.header("Content-Encoding"))) {
                        try (var gzip = new GZIPInputStream(new ByteArrayInputStream(bytes))) {
                            bytes = gzip.readNBytes(16 * 1024);
                        }
                    }
                    body = new String(bytes, StandardCharsets.UTF_8);
                } catch (IOException unreadable) {
                    body = "已收到错误状态，响应内容读取失败";
                }
                throw GenerationProviderFailure.http(response.code(), body);
            }
        });
    }

    private static final class ExchangePermit {
        private final AtomicBoolean used = new AtomicBoolean();
    }

    // 单独关闭连接重试仍不足以阻止 Retry-After: 0 等 HTTP 跟进请求。
    private static final class SingleUseBody extends RequestBody {
        private final RequestBody delegate;

        private SingleUseBody(RequestBody delegate) {
            this.delegate = delegate;
        }

        @Override
        public MediaType contentType() {
            return delegate.contentType();
        }

        @Override
        public long contentLength() throws IOException {
            return delegate.contentLength();
        }

        @Override
        public void writeTo(BufferedSink sink) throws IOException {
            delegate.writeTo(sink);
        }

        @Override
        public boolean isOneShot() {
            return true;
        }

        @Override
        public boolean isDuplex() {
            return delegate.isDuplex();
        }
    }
}
