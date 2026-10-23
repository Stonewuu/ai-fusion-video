package com.stonewu.fusion.service.ai.agentscope.mcp;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.agentscope.core.tool.mcp.McpClientBuilder;

import java.io.IOException;
import java.net.http.HttpRequest;
import java.nio.ByteBuffer;
import java.util.concurrent.Flow;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * 阻止底层 HTTP 客户端在断连后重新发送 tools/call 请求体，协议发现请求不受影响。
 */
public final class McpSingleSubmissionHttp {
    private static final ObjectMapper JSON = new ObjectMapper();

    private McpSingleSubmissionHttp() {
    }

    public static McpClientBuilder configure(McpClientBuilder builder) {
        return builder.httpRequestCustomizer((request, method, uri, body, context) -> {
            if (!"POST".equals(method) || body == null) {
                return;
            }
            try {
                if ("tools/call".equals(JSON.readTree(body).path("method").asText())) {
                    request.POST(new Once(HttpRequest.BodyPublishers.ofString(body)));
                }
            } catch (IOException failure) {
                throw new IllegalArgumentException("MCP 请求无法解析，未发送", failure);
            }
        });
    }

    static final class Once implements HttpRequest.BodyPublisher {
        private final HttpRequest.BodyPublisher delegate;
        private final AtomicBoolean subscribed = new AtomicBoolean();

        Once(HttpRequest.BodyPublisher delegate) {
            this.delegate = delegate;
        }

        @Override
        public long contentLength() {
            return delegate.contentLength();
        }

        @Override
        public void subscribe(Flow.Subscriber<? super ByteBuffer> subscriber) {
            if (subscribed.compareAndSet(false, true)) {
                delegate.subscribe(subscriber);
                return;
            }
            subscriber.onSubscribe(new Flow.Subscription() {
                @Override
                public void request(long n) {
                }

                @Override
                public void cancel() {
                }
            });
            subscriber.onError(new IOException("MCP 原请求体已发送，禁止 HTTP 自动重发"));
        }
    }
}
