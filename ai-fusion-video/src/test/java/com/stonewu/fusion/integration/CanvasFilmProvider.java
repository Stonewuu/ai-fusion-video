package com.stonewu.fusion.integration;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;

/** 有真实可解码字节的本机供应商；记录请求次数，合成和混音仍由产品 FFmpeg 链路执行。 */
final class CanvasFilmProvider implements AutoCloseable {
    private final HttpServer server;
    private final ObjectMapper mapper;
    private final Map<String, byte[]> videos = new ConcurrentHashMap<>();
    final AtomicInteger imagePosts = new AtomicInteger();
    final AtomicInteger videoPosts = new AtomicInteger();
    final AtomicInteger audioPosts = new AtomicInteger();
    private final byte[] red;
    private final byte[] blue;

    CanvasFilmProvider(ObjectMapper mapper, Path directory) throws Exception {
        this.mapper = mapper;
        Files.createDirectories(directory);
        red = png(Color.RED); blue = png(Color.BLUE);
        byte[] redVideo = video(directory, "red"), blueVideo = video(directory, "blue");
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/v1/images/generations", exchange -> {
            var request = mapper.readTree(exchange.getRequestBody());
            imagePosts.incrementAndGet();
            assertThat(request.path("n").asInt()).isEqualTo(2);
            sendJson(exchange, Map.of("data", List.of(Map.of("b64_json", Base64.getEncoder().encodeToString(red)),
                    Map.of("b64_json", Base64.getEncoder().encodeToString(blue)))));
        });
        server.createContext("/v1/videos", exchange -> {
            String path = exchange.getRequestURI().getPath();
            if (path.equals("/v1/videos") && exchange.getRequestMethod().equals("POST")) {
                String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
                assertThat(body).contains("input_reference", "蓝色雨衣");
                String id = "film-" + videoPosts.incrementAndGet();
                videos.put(id, body.contains("镜头 0") ? redVideo : blueVideo);
                sendJson(exchange, Map.of("id", id, "status", "queued"));
                return;
            }
            String id = path.substring("/v1/videos/".length()).split("/")[0];
            if (!videos.containsKey(id)) { exchange.sendResponseHeaders(404, -1); exchange.close(); return; }
            if (path.endsWith("/content")) {
                boolean cover = "variant=thumbnail".equals(exchange.getRequestURI().getQuery());
                send(exchange, cover ? red : videos.get(id), cover ? "image/png" : "video/mp4");
            } else sendJson(exchange, Map.of("id", id, "status", "completed", "seconds", "4"));
        });
        for (String operation : List.of("music", "sfx")) server.createContext("/v1/video-to-" + operation, exchange -> {
            assertThat(exchange.getRequestBody().readAllBytes()).isNotEmpty();
            audioPosts.incrementAndGet(); sendJson(exchange, Map.of("task_id", "film-" + operation));
        });
        server.createContext("/v1/tasks/", exchange -> sendJson(exchange, Map.of("status", "succeeded", "audio", Map.of(
                "url", "data:audio/wav;base64," + Base64.getEncoder().encodeToString(wav(exchange.getRequestURI().getPath().endsWith("sfx") ? 880 : 440))))));
        server.start();
    }
    String url() { return "http://127.0.0.1:" + server.getAddress().getPort(); }
    @Override public void close() { server.stop(0); }
    private void sendJson(HttpExchange exchange, Object value) throws IOException { send(exchange, mapper.writeValueAsBytes(value), "application/json"); }
    private static void send(HttpExchange exchange, byte[] bytes, String contentType) throws IOException {
        exchange.getResponseHeaders().set("Content-Type", contentType); exchange.sendResponseHeaders(200, bytes.length);
        try (var output = exchange.getResponseBody()) { output.write(bytes); }
    }
    private static byte[] png(Color color) throws IOException {
        var image = new BufferedImage(1280, 720, BufferedImage.TYPE_INT_RGB);
        var graphics = image.createGraphics(); graphics.setColor(color); graphics.fillRect(0, 0, 1280, 720); graphics.dispose();
        var output = new ByteArrayOutputStream(); ImageIO.write(image, "png", output); return output.toByteArray();
    }
    private static byte[] video(Path directory, String color) throws Exception {
        Path output = directory.resolve(color + ".mp4");
        execute(directory.resolve(color + ".log"), List.of("ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                "color=c=" + color + ":s=160x90:r=24", "-t", "4", "-c:v", "libx264", "-pix_fmt", "yuv420p", output.toString()));
        return Files.readAllBytes(output);
    }
    private static byte[] wav(int frequency) {
        int length = 48000 * 8 * 2;
        var buffer = ByteBuffer.allocate(44 + length).order(ByteOrder.LITTLE_ENDIAN);
        buffer.put("RIFF".getBytes(StandardCharsets.US_ASCII)).putInt(36 + length).put("WAVEfmt ".getBytes(StandardCharsets.US_ASCII))
                .putInt(16).putShort((short) 1).putShort((short) 1).putInt(48000).putInt(96000).putShort((short) 2).putShort((short) 16)
                .put("data".getBytes(StandardCharsets.US_ASCII)).putInt(length);
        for (int index = 0; index < length / 2; index++) buffer.putShort((short) (Math.sin(index * frequency * 2 * Math.PI / 48000) * 4000));
        return buffer.array();
    }
    static void execute(Path log, List<String> command) throws Exception {
        var process = new ProcessBuilder(new ArrayList<>(command)).redirectErrorStream(true).redirectOutput(log.toFile()).start();
        if (!process.waitFor(30, TimeUnit.SECONDS)) { process.destroyForcibly(); throw new IOException("媒体验收命令超时"); }
        assertThat(process.exitValue()).as(Files.readString(log)).isZero();
    }
}
