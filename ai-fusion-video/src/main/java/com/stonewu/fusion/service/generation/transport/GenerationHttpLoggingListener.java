package com.stonewu.fusion.service.generation.transport;

import lombok.extern.slf4j.Slf4j;
import okhttp3.Call;
import okhttp3.EventListener;
import okhttp3.Request;
import okhttp3.Response;
import org.slf4j.MDC;

import java.io.IOException;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

/** 记录传输阶段和耗时，不读取请求体、响应体、鉴权头或 URL 查询参数。 */
@Slf4j
final class GenerationHttpLoggingListener extends EventListener {
    private final String requestId = UUID.randomUUID().toString();
    private final String jobId = MDC.get("generationJobId");
    private final String method;
    private final String endpoint;
    private long started;

    GenerationHttpLoggingListener(Call call) {
        Request request = call.request();
        method = request.method();
        endpoint = request.url().newBuilder().username("").password("").query(null).fragment(null).build().toString();
    }

    @Override
    public void callStart(Call call) {
        started = System.nanoTime();
        log.debug("生成 HTTP 开始: jobId={}, requestId={}, method={}, endpoint={}", jobId, requestId, method, endpoint);
    }

    @Override
    public void requestHeadersEnd(Call call, Request request) {
        log.debug("生成 HTTP 请求已发送: jobId={}, requestId={}, elapsedMs={}", jobId, requestId, elapsedMillis());
    }

    @Override
    public void responseHeadersEnd(Call call, Response response) {
        if (response.code() >= 400) {
            log.error("生成 HTTP 返回错误: jobId={}, requestId={}, status={}, elapsedMs={}",
                    jobId, requestId, response.code(), elapsedMillis());
        } else {
            log.debug("生成 HTTP 已收到响应头: jobId={}, requestId={}, status={}, elapsedMs={}",
                    jobId, requestId, response.code(), elapsedMillis());
        }
    }

    @Override
    public void responseBodyEnd(Call call, long byteCount) {
        log.debug("生成 HTTP 响应接收完成: jobId={}, requestId={}, bytes={}, elapsedMs={}",
                jobId, requestId, byteCount, elapsedMillis());
    }

    @Override
    public void callEnd(Call call) {
        log.debug("生成 HTTP 完成: jobId={}, requestId={}, elapsedMs={}", jobId, requestId, elapsedMillis());
    }

    @Override
    public void callFailed(Call call, IOException failure) {
        log.error("生成 HTTP 连接失败: jobId={}, requestId={}, method={}, endpoint={}, elapsedMs={}, errorType={}",
                jobId, requestId, method, endpoint, elapsedMillis(), failure.getClass().getSimpleName());
    }

    private long elapsedMillis() {
        return TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - started);
    }
}
