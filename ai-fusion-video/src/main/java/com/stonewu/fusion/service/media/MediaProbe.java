package com.stonewu.fusion.service.media;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaFacts;
import com.stonewu.fusion.service.media.model.MediaKind;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.TimeUnit;

/**
 * 先检查内容，再登记类型；扩展名和浏览器 MIME 不作为媒体事实。
 */
@Component
public class MediaProbe {
    private static final Set<String> TEXT_TYPES = Set.of("text/plain", "text/markdown", "text/csv", "application/json");
    private final ObjectMapper mapper;
    private final String ffprobe;

    public MediaProbe(ObjectMapper mapper, @Value("${video.compose.ffprobe-path:ffprobe}") String ffprobe) {
        this.mapper = mapper;
        this.ffprobe = ffprobe;
    }

    public MediaFacts inspect(Path file, String declaredMime) throws IOException {
        long size = Files.size(file);
        if (size == 0) {
            throw MediaException.invalid("不能上传空文件");
        }
        String hash = digest(file);
        byte[] prefix;
        try (var input = Files.newInputStream(file)) {
            prefix = input.readNBytes(4096);
        }
        if (ascii(prefix).startsWith("%PDF-")) {
            if (size < 12) {
                throw MediaException.invalid("PDF 文件不完整");
            }
            try (var input = Files.newByteChannel(file)) {
                input.position(Math.max(0, size - 2048));
                ByteBuffer tail = ByteBuffer.allocate(2048);
                input.read(tail);
                String text = new String(tail.array(), 0, tail.position(), StandardCharsets.ISO_8859_1);
                if (!text.contains("%%EOF")) {
                    throw MediaException.invalid("PDF 文件缺少结束标记");
                }
            }
            return new MediaFacts(MediaKind.DOCUMENT, "application/pdf", size, hash, null, null, null, false);
        }
        String declared = declaredMime == null ? "" : declaredMime.toLowerCase(Locale.ROOT).split(";", 2)[0].trim();
        if (TEXT_TYPES.contains(declared)) {
            return text(file, declared, size, hash);
        }
        String imageMime = imageMime(prefix);
        JsonNode probe = probe(file);
        JsonNode visual = null;
        boolean audio = false;
        long duration = millis(probe.path("format").path("duration").asText(""));
        for (JsonNode stream : probe.path("streams")) {
            if ("audio".equals(stream.path("codec_type").asText())) {
                audio = true;
            }
            if ("video".equals(stream.path("codec_type").asText()) && stream.path("disposition").path("attached_pic")
                    .asInt() == 0) {
                if (visual == null) {
                    visual = stream;
                }
            }
            duration = Math.max(duration, millis(stream.path("duration").asText("")));
        }
        Integer width = visual == null ? null : visual.path("width").asInt();
        Integer height = visual == null ? null : visual.path("height").asInt();
        if (visual != null && (width <= 0 || height <= 0 || width > 32768 || height > 32768
                || (long) width * height > 268435456L)) {
            throw MediaException.invalid("媒体画面尺寸无效或超过限制");
        }
        if (imageMime != null) {
            if (visual == null || audio) {
                throw MediaException.invalid("图片内容与文件标识不符");
            }
            return new MediaFacts(MediaKind.IMAGE, imageMime, size, hash, width, height, null, false);
        }
        String format = probe.path("format").path("format_name").asText("");
        MediaKind kind = visual != null ? MediaKind.VIDEO : MediaKind.AUDIO;
        if ((visual == null && !audio) || duration <= 0) {
            throw MediaException.invalid("无法取得媒体的有效音视频流和时长");
        }
        String mime = MediaContainerMime.resolve(format, kind, prefix);
        return new MediaFacts(kind, mime, size, hash, width, height, duration, audio);
    }

    private MediaFacts text(Path file, String mime, long size, String hash) throws IOException {
        if (size > 10 * 1024 * 1024) {
            throw MediaException.invalid("文本文件不能超过 10 MB");
        }
        String text;
        try {
            text = StandardCharsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
                    .onUnmappableCharacter(CodingErrorAction.REPORT).decode(ByteBuffer.wrap(Files.readAllBytes(file)))
                    .toString();
        } catch (CharacterCodingException invalid) {
            throw MediaException.invalid("文本文件必须使用有效的 UTF-8 编码");
        }
        if (text.codePoints().anyMatch(point -> point < 32 && point != 9 && point != 10 && point != 13)) {
            throw MediaException.invalid("文本文件包含不支持的控制字符");
        }
        if ("application/json".equals(mime)) {
            try (var parser = mapper.createParser(text)) {
                if (mapper.readTree(parser) == null || parser.nextToken() != null) {
                    throw MediaException.invalid("JSON 文件格式无效");
                }
            } catch (IOException invalid) {
                throw MediaException.invalid("JSON 文件格式无效");
            }
        }
        return new MediaFacts(MediaKind.TEXT, mime, size, hash, null, null, null, false);
    }

    private JsonNode probe(Path file) throws IOException {
        Path output = Files.createTempFile("afv-probe-", ".json");
        Process process = null;
        try {
            ProcessBuilder builder = new ProcessBuilder(ffprobe, "-v", "error", "-protocol_whitelist", "file,pipe",
                    "-probesize", "10000000", "-analyzeduration", "10000000",
                    "-show_entries",
                    "format=format_name,duration:stream=codec_type,width,height,duration:stream_disposition=attached_pic",
                    "-of", "json", file.toAbsolutePath().toString());
            builder.redirectOutput(output.toFile()).redirectError(ProcessBuilder.Redirect.DISCARD);
            try {
                process = builder.start();
            } catch (IOException missing) {
                throw MediaException.unavailable("媒体检测程序不可用，请配置 video.compose.ffprobe-path");
            }
            process.getOutputStream().close();
            if (!process.waitFor(30, TimeUnit.SECONDS)) {
                throw MediaException.invalid("媒体检测超时");
            }
            if (process.exitValue() != 0 || Files.size(output) > 256 * 1024) {
                throw MediaException.invalid("媒体内容损坏或格式不受支持");
            }
            return mapper.readTree(Files.readString(output));
        } catch (InterruptedException cancelled) {
            Thread.currentThread().interrupt();
            throw new IOException("媒体检测已中断", cancelled);
        } finally {
            if (process != null && process.isAlive()) {
                process.destroyForcibly();
                try {
                    process.waitFor(5, TimeUnit.SECONDS);
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                }
            }
            Files.deleteIfExists(output);
        }
    }

    private String imageMime(byte[] bytes) {
        if (bytes.length >= 8 && HexFormat.of().formatHex(bytes, 0, 8).equals("89504e470d0a1a0a")) {
            return "image/png";
        }
        if (bytes.length >= 3 && (bytes[0] & 255) == 255 && (bytes[1] & 255) == 216 && (bytes[2] & 255) == 255) {
            return "image/jpeg";
        }
        String text = ascii(bytes);
        if (text.startsWith("GIF87a") || text.startsWith("GIF89a")) {
            return "image/gif";
        }
        if (text.startsWith("RIFF") && text.length() >= 12 && text.substring(8, 12).equals("WEBP")) {
            return "image/webp";
        }
        return null;
    }

    private long millis(String raw) {
        try {
            double seconds = Double.parseDouble(raw);
            if (!Double.isFinite(seconds) || seconds <= 0 || seconds > 604800) {
                return 0;
            }
            return Math.max(1, Math.round(seconds * 1000));
        } catch (NumberFormatException invalid) {
            return 0;
        }
    }

    private String digest(Path file) throws IOException {
        try {
            var digest = MessageDigest.getInstance("SHA-256");
            try (var input = Files.newInputStream(file)) {
                byte[] buffer = new byte[65536];
                for (int count; (count = input.read(buffer)) != -1; ) {
                    digest.update(buffer, 0, count);
                }
            }
            return HexFormat.of().formatHex(digest.digest());
        } catch (NoSuchAlgorithmException unavailable) {
            throw new IllegalStateException(unavailable);
        }
    }

    private String ascii(byte[] bytes) {
        return new String(bytes, StandardCharsets.ISO_8859_1);
    }
}
