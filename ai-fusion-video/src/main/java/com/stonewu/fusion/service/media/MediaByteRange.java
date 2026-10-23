package com.stonewu.fusion.service.media;

import com.stonewu.fusion.service.media.model.MediaException;

/**
 * 单范围用于浏览器音视频 seek；不把多范围请求合并为意外的整文件下载。
 */
public record MediaByteRange(long start, long end, boolean partial) {
    public long length() {
        return end - start + 1;
    }

    public static MediaByteRange parse(String value, long size) {
        if (size <= 0) {
            throw invalid();
        }
        if (value == null) {
            return new MediaByteRange(0, size - 1, false);
        }
        if (!value.matches("bytes=[0-9]*-[0-9]*")) {
            throw invalid();
        }
        String[] parts = value.substring(6).split("-", -1);
        try {
            long start;
            long end;
            if (parts[0].isEmpty()) {
                long suffix = Long.parseLong(parts[1]);
                if (suffix <= 0) {
                    throw invalid();
                }
                start = suffix >= size ? 0 : size - suffix;
                end = size - 1;
            } else {
                start = Long.parseLong(parts[0]);
                end = parts[1].isEmpty() ? size - 1 : Math.min(Long.parseLong(parts[1]), size - 1);
            }
            if (start < 0 || start >= size || end < start) {
                throw invalid();
            }
            return new MediaByteRange(start, end, true);
        } catch (NumberFormatException invalid) {
            throw invalid();
        }
    }

    private static MediaException invalid() {
        return new MediaException(416, "INVALID_MEDIA_RANGE", "媒体读取范围无效");
    }
}
