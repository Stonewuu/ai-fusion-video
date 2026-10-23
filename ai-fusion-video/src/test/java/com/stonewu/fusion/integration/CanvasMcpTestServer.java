package com.stonewu.fusion.integration;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import io.modelcontextprotocol.spec.McpSchema;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/** 真实 MCP HTTP 协议夹具；只有工具 POST 才计数，可注入丢回执和延迟回执。 */
final class CanvasMcpTestServer implements AutoCloseable {
    record Scenario(String mode, AtomicInteger calls, CountDownLatch entered, CountDownLatch release) { }
    private final ObjectMapper mapper = new ObjectMapper();
    private final HttpServer server;
    private final ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
    private final Map<String, Scenario> scenarios = new ConcurrentHashMap<>();
    private final Map<String, JsonNode> requests = new ConcurrentHashMap<>();
    private volatile boolean revised;
    CanvasMcpTestServer() {
        try {
            server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
            server.setExecutor(executor); server.createContext("/mcp", this::handle); server.start();
        } catch (IOException failure) { throw new IllegalStateException(failure); }
    }
    String url() { return "http://127.0.0.1:" + server.getAddress().getPort() + "/mcp"; }
    Scenario scenario(String key, String mode) {
        var value = new Scenario(mode, new AtomicInteger(), new CountDownLatch(1), new CountDownLatch(1));
        scenarios.put(key, value); return value;
    }
    JsonNode request(String key) { return requests.get(key); }
    void reviseDefinition() { revised = true; }
    McpSchema.Tool tool() {
        try { return mapper.readValue(toolJson(), McpSchema.Tool.class); }
        catch (IOException failure) { throw new IllegalStateException(failure); }
    }
    private String toolJson() {
        String source = """
                {"name":"inspect_text","description":"本机协议验收，返回指定文本",
                 "inputSchema":{"type":"object","properties":{"text":{"type":"string"},"requestKey":{"type":"string"}},"required":["text"],"additionalProperties":false},
                 "outputSchema":{"type":"object","properties":{"result":{"type":"string"}},"required":["result"]},
                 "annotations":{"readOnlyHint":true,"idempotentHint":true,"destructiveHint":false}}
                """;
        if (!revised) return source;
        try {
            var schema = mapper.readTree(source);
            var properties = (ObjectNode) schema.at("/inputSchema/properties");
            properties.set("language", mapper.valueToTree(Map.of("type", "string")));
            return schema.toString();
        } catch (IOException failure) { throw new IllegalStateException(failure); }
    }
    private void handle(HttpExchange exchange) {
        try {
            if (!exchange.getRequestMethod().equals("POST")) { exchange.sendResponseHeaders(405, -1); return; }
            var body = mapper.readTree(exchange.getRequestBody().readAllBytes());
            if (body == null) { exchange.sendResponseHeaders(400, -1); return; }
            if (!body.has("id")) { exchange.sendResponseHeaders(202, -1); return; }
            Object result;
            switch (body.path("method").asText()) {
                case "initialize" -> result = Map.of("protocolVersion", body.at("/params/protocolVersion").asText(),
                        "capabilities", Map.of("tools", Map.of()), "serverInfo", Map.of("name", "canvas-mcp-test", "version", "1"));
                case "tools/list" -> result = Map.of("tools", List.of(mapper.readTree(toolJson())));
                case "tools/call" -> {
                    String text = body.at("/params/arguments/text").asText();
                    var scenario = scenarios.get(text);
                    if (scenario == null) throw new IllegalStateException("未登记的测试调用 " + text);
                    scenario.calls().incrementAndGet(); requests.put(text, body); scenario.entered().countDown();
                    if (scenario.mode().equals("drop")) return;
                    if (scenario.mode().equals("hold") && !scenario.release().await(25, TimeUnit.SECONDS)) throw new IllegalStateException("测试回执未释放");
                    result = Map.of("content", List.of(Map.of("type", "text", "text", "读取完成")),
                            "structuredContent", Map.of("result", "已读取：" + text), "isError", false);
                }
                default -> { reply(exchange, Map.of("jsonrpc", "2.0", "id", body.get("id"), "error", Map.of("code", -32601, "message", "unsupported"))); return; }
            }
            reply(exchange, Map.of("jsonrpc", "2.0", "id", body.get("id"), "result", result));
        } catch (IOException ignored) {
            // 请求体被单次发送保护拒绝或客户端已经停止等待。
        } catch (InterruptedException interrupted) { Thread.currentThread().interrupt(); }
        finally { exchange.close(); }
    }
    private void reply(HttpExchange exchange, Object value) throws IOException {
        byte[] bytes = mapper.writeValueAsBytes(value);
        exchange.getResponseHeaders().add("Content-Type", "application/json");
        exchange.sendResponseHeaders(200, bytes.length);
        exchange.getResponseBody().write(bytes);
    }
    @Override public void close() {
        scenarios.values().forEach(value -> value.release().countDown());
        server.stop(0); executor.shutdownNow();
    }
}
