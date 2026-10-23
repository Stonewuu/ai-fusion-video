package com.stonewu.fusion.service.media.preview;

import com.stonewu.fusion.service.media.model.MediaResource;
import jakarta.validation.constraints.Pattern;

public final class MediaPreviewData {
    private MediaPreviewData() {
    }

    public static final String KIND = "MEDIA_PREVIEW";
    public static final String PROFILE = "canvas_640_v1";
    public static final int WIDTH = 640;
    public static final int HEIGHT = 480;
    public static final long MAX_PIXELS = 64_000_000;

    public record Request(@Pattern(regexp = "[0-9a-f-]{36}") String retryJobId) {
    }

    public record Input(int schemaVersion, MediaResource source, String profile, long sampleAtMs) {
    }

    public record View(String sourceId, int sourceVersion, String profile, String jobId,
                       String status, String error, String previewId, Integer width, Integer height,
                       long sampleAtMs, boolean canRetry) {
    }

    public record Parent(String id, int version) {
    }
}
