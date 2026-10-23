package com.stonewu.fusion.service.media.model;

/**
 * 由内容探测获得，不能信任上传者提供的文件类型、尺寸或时长。
 */
public record MediaFacts(MediaKind kind, String mimeType, long byteSize, String contentHash,
                         Integer width, Integer height, Long durationMs, boolean hasAudio) {
}
