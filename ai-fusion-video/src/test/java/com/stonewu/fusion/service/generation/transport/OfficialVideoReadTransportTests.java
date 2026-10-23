package com.stonewu.fusion.service.generation.transport;

import cn.hutool.json.JSONObject;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;

import java.io.UncheckedIOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class OfficialVideoReadTransportTests {
    /** 空闲连接断开时只读发现能够重连，但同一客户端的生成 POST 不能被重复提交。 */
    @Test
    void readCanRecoverLostConnectionWhileGenerationPostRemainsSingleUse() throws Exception {
        var reads = new AtomicInteger();
        var posts = new AtomicInteger();
        var server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/", exchange -> {
            exchange.getRequestBody().readAllBytes();
            if (exchange.getRequestMethod().equals("POST")) {
                posts.incrementAndGet();
                exchange.close();
                return;
            }
            if (exchange.getRequestURI().getPath().equals("/read") && reads.incrementAndGet() == 1) {
                exchange.close();
                return;
            }
            byte[] body = "{\"balance\":20}".getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().set("Content-Type", "application/json");
            exchange.sendResponseHeaders(200, body.length);
            exchange.getResponseBody().write(body);
            exchange.close();
        });
        server.start();
        try {
            String base = "http://127.0.0.1:" + server.getAddress().getPort();
            var client = new OfficialVideoHttpClient(ApiConfig.builder().apiKey("fixture-key").apiUrl(base).build());
            client.get(base + "/warmup");
            assertThat(client.get(base + "/read").getInt("balance")).isEqualTo(20);
            assertThat(reads).hasValue(2);
            var submit = client.post(base + "/submit", new JSONObject().set("prompt", "fixture"), Map.of());
            assertThatThrownBy(() -> client.execute(submit)).isInstanceOf(UncheckedIOException.class);
            assertThat(posts).hasValue(1);
        } finally {
            server.stop(0);
        }
    }
}
