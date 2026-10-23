package com.stonewu.fusion.service.media;

import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaKind;

import java.nio.charset.StandardCharsets;
import java.util.Set;

/**
 * ffprobe 对 QuickTime/MP4、Matroska/WebM 返回合并名称，因此再读取容器内的明确格式标识。
 */
final class MediaContainerMime {
    private MediaContainerMime() {
    }

    static String resolve(String formats, MediaKind kind, byte[] prefix) {
        Set<String> names = Set.of(formats.split(","));
        boolean video = kind == MediaKind.VIDEO;
        if (names.contains("mov") || names.contains("mp4")) {
            if (prefix.length >= 12 && ascii(prefix, 4, 4).equals("ftyp")) {
                String brand = ascii(prefix, 8, 4);
                if (brand.equals("qt  ")) {
                    return video ? "video/quicktime" : "audio/mp4";
                }
                if (brand.startsWith("3gp")) {
                    return video ? "video/3gpp" : "audio/3gpp";
                }
                if (brand.startsWith("3g2")) {
                    return video ? "video/3gpp2" : "audio/3gpp2";
                }
                return video ? "video/mp4" : "audio/mp4";
            }
            // 无 ftyp 的传统 QuickTime 文件由 moov/mdat 等原子组成。
            if (prefix.length >= 8 && Set.of("moov", "mdat", "wide", "free").contains(ascii(prefix, 4, 4))) {
                return video ? "video/quicktime" : "audio/mp4";
            }
            throw MediaException.invalid("音视频容器缺少有效的格式标识");
        }
        if (names.contains("webm") || names.contains("matroska")) {
            String type = ebmlDocType(prefix);
            if ("webm".equals(type)) {
                return video ? "video/webm" : "audio/webm";
            }
            if ("matroska".equals(type)) {
                return video ? "video/x-matroska" : "audio/x-matroska";
            }
            throw MediaException.invalid("不支持此 EBML 文档类型");
        }
        if (names.contains("mp3") && !video) {
            return "audio/mpeg";
        }
        if (names.contains("wav") && !video) {
            return "audio/wav";
        }
        if (names.contains("flac") && !video) {
            return "audio/flac";
        }
        if (names.contains("ogg")) {
            return video ? "video/ogg" : "audio/ogg";
        }
        if (names.contains("aac") && !video) {
            return "audio/aac";
        }
        if (names.contains("mpeg") && video) {
            return "video/mpeg";
        }
        throw MediaException.invalid("不支持此媒体容器格式");
    }

    private static String ebmlDocType(byte[] bytes) {
        if (bytes.length < 5 || bytes[0] != 0x1a || bytes[1] != 0x45 || (bytes[2] & 255) != 0xdf || (bytes[3] & 255) != 0xa3) {
            throw MediaException.invalid("EBML 文件头无效");
        }
        Vint header = vint(bytes, 4, false);
        int offset = 4 + header.length();
        long end = offset + header.value();
        if (end > bytes.length) {
            throw MediaException.invalid("EBML 文件头过大或不完整");
        }
        while (offset < end) {
            Vint id = vint(bytes, offset, true);
            offset += id.length();
            Vint size = vint(bytes, offset, false);
            offset += size.length();
            if (size.value() > end - offset) {
                throw MediaException.invalid("EBML 文件头不完整");
            }
            if (id.value() == 0x4282) {
                return ascii(bytes, offset, (int) size.value());
            }
            offset += (int) size.value();
        }
        throw MediaException.invalid("EBML 文件缺少文档类型");
    }

    private static Vint vint(byte[] data, int offset, boolean identifier) {
        if (offset >= data.length) {
            throw MediaException.invalid("EBML 文件头不完整");
        }
        int first = data[offset] & 255;
        int length = 1;
        int marker = 128;
        while (length <= 8 && (first & marker) == 0) {
            length++;
            marker >>= 1;
        }
        if (length > 8 || offset + length > data.length) {
            throw MediaException.invalid("EBML 长度格式无效");
        }
        long value = identifier ? first : first & (marker - 1);
        for (int index = 1; index < length; index++) {
            value = (value << 8) | (data[offset + index] & 255);
        }
        return new Vint(length, value);
    }

    private static String ascii(byte[] data, int offset, int length) {
        return new String(data, offset, length, StandardCharsets.ISO_8859_1);
    }

    private record Vint(int length, long value) {
    }
}
