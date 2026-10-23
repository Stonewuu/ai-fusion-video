package com.stonewu.fusion.service.canvas.delivery;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.canvas.model.CanvasGraph;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Value;
import com.stonewu.fusion.service.media.model.MediaResource;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;

import java.time.Instant;
import java.util.List;
import java.util.Map;

public final class CanvasExportData {
    private CanvasExportData() {
    }

    public static final long MAX_BYTES = 4L * 1024 * 1024 * 1024;

    public enum Format {MARKDOWN, CSV, JSON, ZIP, MEDIA}

    public enum Scope {CANVAS, SELECTION, PROJECT}

    public record Options(@NotNull Format format, @NotNull Scope scope, @NotNull List<String> nodeIds,
                          @NotNull Long expectedRevision) {
    }

    public record Submit(@NotBlank @Pattern(regexp = "[A-Za-z0-9:._-]{1,96}") String requestId,
                         @NotNull @Valid Options options, @NotBlank String previewHash) {
    }

    public record Manifest(int schemaVersion, String title, long documentRevision, CanvasGraph graph,
                           List<Value> values, List<MediaResource> media, List<String> warnings,
                           @JsonInclude(JsonInclude.Include.NON_EMPTY) Map<String, Value> fixedOutputs) {
        public Manifest {
            // 已持久化的第一版交付没有此字段，保持旧清单的序列化和摘要不变。
            if (fixedOutputs == null) {
                fixedOutputs = Map.of();
            }
            if (schemaVersion < 1 || schemaVersion > 2) {
                throw invalid("不支持的交付快照协议");
            }
            fixedOutputs = Map.copyOf(fixedOutputs);
        }
    }

    public record Preview(String hash, long documentRevision, int nodes, int businessObjects, int mediaFiles,
                          long mediaBytes, List<String> warnings) {
    }

    public record Job(String id, String canvasId, String nodeRunId, Format format, String status,
                      ResourceCommandContext context,
                      Manifest manifest, String manifestHash, boolean cancelRequested, long revision,
                      int attempts, String worker, long epoch, Instant leaseUntil, String blobId,
                      String filename, String mimeType, Long byteSize, String contentHash, String error,
                      Instant createdAt, Instant updatedAt) {
        public boolean terminal() {
            return List.of("SUCCEEDED", "FAILED", "CANCELLED").contains(status);
        }
    }

    public record Summary(String id, String canvasId, Format format, String status, String filename, Long byteSize,
                          String contentHash, long revision, int attempts, String error, Instant createdAt,
                          Instant updatedAt, String actorName, boolean canCancel, int mediaFiles, int businessObjects,
                          boolean contentUnavailable) {
    }

    public record Claim(String id, String worker, long epoch) {
    }

    public static BusinessException invalid(String message) {
        return new BusinessException(400, message);
    }

    public static BusinessException conflict(String message) {
        return new BusinessException(409, message);
    }
}
