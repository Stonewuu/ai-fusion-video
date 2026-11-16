package com.stonewu.fusion.service.ai.comfyui;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import com.stonewu.fusion.service.media.MediaIntakeService;
import okhttp3.OkHttpClient;
import okhttp3.Request;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Base64;
import java.util.Locale;

/** 音视频输入经临时文件传输，避免下载和上传时各复制一份完整媒体到内存。 */
record ComfyUiMediaInputFile(Path path, String contentType, String extension) implements AutoCloseable {
    static ComfyUiMediaInputFile load(ApiConfig apiConfig, String source, String mediaType, OkHttpClient client)
            throws IOException {
        if (source == null || source.isBlank()) {
            throw new BusinessException(400, "ComfyUI 媒体输入不能为空");
        }
        String value = source.trim();
        if (value.startsWith("data:")) {
            return dataUri(value, mediaType);
        }
        return download(apiConfig, value, mediaType, client);
    }

    private static ComfyUiMediaInputFile dataUri(String source, String mediaType) throws IOException {
        int comma = source.indexOf(',');
        if (comma <= 5) {
            throw new BusinessException(400, "ComfyUI 媒体 Data URI 格式无效");
        }
        String[] metadata = source.substring(5, comma).split(";");
        if (Arrays.stream(metadata).skip(1).noneMatch(part -> part.trim().equalsIgnoreCase("base64"))) {
            throw new BusinessException(400, "ComfyUI 媒体 Data URI 必须使用 base64");
        }
        String contentType = metadata[0].trim().toLowerCase(Locale.ROOT);
        String extension = extension(contentType, mediaType);
        if (source.length() - comma - 1 > (MediaIntakeService.MAX_UPLOAD_BYTES + 2) / 3 * 4) {
            throw tooLarge(true);
        }
        Path temporary = Files.createTempFile("comfyui-input-", "." + extension);
        boolean ready = false;
        var encoded = new Base64TextInputStream(source, comma + 1);
        try (InputStream decoded = Base64.getDecoder().wrap(encoded)) {
            copy(decoded, temporary, MediaIntakeService.MAX_UPLOAD_BYTES, true);
            if (encoded.position != source.length()) {
                throw new BusinessException(400, "ComfyUI 媒体 Data URI base64 无效");
            }
            ready = true;
            return new ComfyUiMediaInputFile(temporary, contentType, extension);
        } finally {
            if (!ready) {
                Files.deleteIfExists(temporary);
            }
        }
    }

    private static ComfyUiMediaInputFile download(ApiConfig config, String source, String mediaType,
                                                  OkHttpClient baseClient) throws IOException {
        URI uri;
        try {
            uri = URI.create(source);
        } catch (IllegalArgumentException failure) {
            throw new BusinessException(400, "ComfyUI 媒体 URL 无效");
        }
        if (uri.getHost() == null || (!"http".equalsIgnoreCase(uri.getScheme())
                && !"https".equalsIgnoreCase(uri.getScheme()))) {
            throw new BusinessException(400, "ComfyUI 媒体输入必须是 HTTP(S) URL 或 Data URI");
        }
        var request = new Request.Builder().url(source).header("Accept", mediaType + "/*").get().build();
        try (var response = AiProxySupport.okHttpClient(baseClient, config).newCall(request).execute()) {
            if (!response.isSuccessful() || response.body() == null) {
                throw new BusinessException(502, "下载 ComfyUI 输入媒体失败，HTTP " + response.code());
            }
            if (response.body().contentLength() > MediaIntakeService.MAX_RESULT_BYTES) {
                throw tooLarge(false);
            }
            if (response.body().contentType() == null) {
                throw new BusinessException(502, "ComfyUI 媒体响应缺少 Content-Type");
            }
            String contentType = response.body().contentType().toString().split(";", 2)[0].toLowerCase(Locale.ROOT);
            String extension = extension(contentType, mediaType);
            Path temporary = Files.createTempFile("comfyui-input-", "." + extension);
            boolean ready = false;
            try (var input = response.body().byteStream()) {
                copy(input, temporary, MediaIntakeService.MAX_RESULT_BYTES, false);
                ready = true;
                return new ComfyUiMediaInputFile(temporary, contentType, extension);
            } finally {
                if (!ready) {
                    Files.deleteIfExists(temporary);
                }
            }
        }
    }

    private static String extension(String contentType, String mediaType) {
        if (!contentType.startsWith(mediaType + "/")) {
            throw new BusinessException(400, "ComfyUI 输入文件类型与绑定不匹配");
        }
        return switch (contentType) {
            case "audio/mpeg", "audio/mp3" -> "mp3";
            case "audio/wav", "audio/x-wav", "audio/wave" -> "wav";
            case "audio/flac", "audio/x-flac" -> "flac";
            case "audio/ogg", "video/ogg" -> "ogg";
            case "audio/opus" -> "opus";
            case "audio/aac" -> "aac";
            case "audio/mp4", "audio/x-m4a" -> "m4a";
            case "audio/webm", "video/webm" -> "webm";
            case "video/mp4" -> "mp4";
            case "video/quicktime" -> "mov";
            case "video/x-matroska", "audio/x-matroska" -> "mkv";
            case "video/x-msvideo" -> "avi";
            case "video/mpeg" -> "mpeg";
            default -> throw new BusinessException(400, "ComfyUI 输入媒体格式不支持：" + contentType);
        };
    }

    private static void copy(InputStream input, Path output, long maximum, boolean dataUri) throws IOException {
        long total = 0;
        byte[] buffer = new byte[65536];
        try (var destination = Files.newOutputStream(output)) {
            while (true) {
                if (Thread.currentThread().isInterrupted()) {
                    throw new IOException("媒体输入传输已中断");
                }
                int count;
                try {
                    count = input.read(buffer);
                } catch (IOException failure) {
                    if (dataUri) {
                        throw new BusinessException(400, "ComfyUI 媒体 Data URI base64 无效");
                    }
                    throw failure;
                }
                if (count == -1) {
                    break;
                }
                total += count;
                if (total > maximum) {
                    throw tooLarge(dataUri);
                }
                destination.write(buffer, 0, count);
            }
        }
        if (total == 0) {
            throw new BusinessException(400, "ComfyUI 输入媒体文件为空");
        }
    }

    private static BusinessException tooLarge(boolean dataUri) {
        return new BusinessException(400, dataUri ? "ComfyUI 媒体 Data URI 超过 100MB" : "ComfyUI 输入媒体超过 2GB");
    }

    @Override
    public void close() throws IOException {
        Files.deleteIfExists(path);
    }

    private static final class Base64TextInputStream extends InputStream {
        private final String source;
        private int position;

        private Base64TextInputStream(String source, int position) {
            this.source = source;
            this.position = position;
        }

        @Override
        public int read() throws IOException {
            if (position == source.length()) {
                return -1;
            }
            char value = source.charAt(position++);
            if (value > 127) {
                throw new IOException("媒体 Data URI 含无效 Base64 字符");
            }
            return value;
        }
    }
}
