package com.stonewu.fusion.integration;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/** 本地增量模型，不访问付费服务；父/子 agent 通过现有工具定义完成真实委派。 */
final class AgentStreamingProvider implements AutoCloseable {
    private final HttpServer server;
    private final ObjectMapper json;
    final AtomicInteger requests = new AtomicInteger();
    private final AtomicInteger delegations = new AtomicInteger();
    volatile CountDownLatch release = new CountDownLatch(1);
    volatile long firstTextNanos;
    volatile boolean gate;
    volatile int chunks = 20;
    volatile long chunkDelayMs = 100;
    volatile String toolName;
    volatile String toolArguments;
    volatile int children = 1;
    volatile boolean holdLastChild;
    final AtomicInteger childRequests = new AtomicInteger();
    final CountDownLatch lastChildRelease = new CountDownLatch(1);

    AgentStreamingProvider(ObjectMapper json) throws IOException {
        this.json = json;
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.setExecutor(Executors.newVirtualThreadPerTaskExecutor());
        server.createContext("/v1/chat/completions", exchange -> {
            requests.incrementAndGet();
            JsonNode request = json.readTree(exchange.getRequestBody());
            if (!request.path("stream").asBoolean()) {
                exchange.sendResponseHeaders(400, -1);
                exchange.close();
                return;
            }
            boolean parent = false;
            for (JsonNode tool : request.path("tools")) {
                if ("episode_scene_writer".equals(tool.path("function").path("name").asText())) parent = true;
            }
            boolean hasResult = false;
            for (JsonNode message : request.path("messages")) {
                if ("user".equals(message.path("role").asText())) {
                    hasResult = false;
                } else if ("tool".equals(message.path("role").asText())) {
                    hasResult = true;
                }
            }
            exchange.getResponseHeaders().set("Content-Type", "text/event-stream");
            exchange.sendResponseHeaders(200, 0);
            try (var output = exchange.getResponseBody()) {
                chunk(output, Map.of("reasoning_content", "检查输入并逐段组织回复。"), null);
                firstTextNanos = System.nanoTime();
                chunk(output, Map.of("content", "首段已到达。\n\n"), null);
                if (gate && !release.await(15, TimeUnit.SECONDS)) throw new IOException("消费者未收到首段");
                if (holdLastChild && !parent && childRequests.incrementAndGet() == 3
                        && !lastChildRelease.await(15, TimeUnit.SECONDS)) {
                    throw new IOException("最后一个子任务未结束验收等待");
                }
                if (toolName != null && !hasResult) {
                    chunk(output, Map.of("tool_calls", List.of(Map.of("index", 0, "id", "write-1", "type", "function",
                            "function", Map.of("name", toolName, "arguments", toolArguments)))), "tool_calls");
                } else if (parent && !hasResult) {
                    int delegation = delegations.incrementAndGet();
                    for (int child = 0; child < children; child++) {
                        String callId = "delegate-" + (child + 1) + (delegation == 1 ? "" : "-batch-" + delegation);
                        chunk(output, Map.of("tool_calls", List.of(Map.of("index", child, "id", callId, "type", "function",
                                "function", Map.of("name", "episode_scene_writer", "arguments", "{\"scriptEpisodeId\":")))), null);
                        chunk(output, Map.of("tool_calls", List.of(Map.of("index", child, "function", Map.of("arguments", "730001}")))), null);
                    }
                    chunk(output, Map.of(), "tool_calls");
                } else {
                    for (int i = 1; i <= chunks; i++) {
                        chunk(output, Map.of("content", "第" + i + "段内容正在连续显示。\n\n"), null);
                        Thread.sleep(chunkDelayMs);
                    }
                    chunk(output, Map.of(), "stop");
                }
                write(output, Map.of("id", "stream-test", "object", "chat.completion.chunk", "choices", List.of(),
                        "usage", Map.of("prompt_tokens", 10, "completion_tokens", 20, "total_tokens", 30)));
                output.write("data: [DONE]\n\n".getBytes(StandardCharsets.UTF_8));
                output.flush();
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
            } finally {
                exchange.close();
            }
        });
        server.start();
    }

    String url() { return "http://127.0.0.1:" + server.getAddress().getPort(); }

    private void chunk(OutputStream output, Map<String, Object> delta, String finish) throws IOException {
        ObjectNode choice = json.createObjectNode().put("index", 0);
        choice.set("delta", json.valueToTree(delta));
        if (finish != null) choice.put("finish_reason", finish);
        write(output, Map.of("id", "stream-test", "object", "chat.completion.chunk", "choices", List.of(choice)));
    }

    private void write(OutputStream output, Object value) throws IOException {
        output.write(("data: " + json.writeValueAsString(value) + "\n\n").getBytes(StandardCharsets.UTF_8));
        output.flush();
    }

    @Override public void close() {
        release.countDown();
        lastChildRelease.countDown();
        server.stop(0);
    }
}
