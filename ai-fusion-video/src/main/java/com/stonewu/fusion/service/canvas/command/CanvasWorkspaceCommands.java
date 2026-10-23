package com.stonewu.fusion.service.canvas.command;

import jakarta.validation.constraints.*;

public final class CanvasWorkspaceCommands {
    private CanvasWorkspaceCommands() {
    }

    public record DocumentPatch(
            @NotBlank @Size(max = 128) @Pattern(regexp = "[A-Za-z0-9._:-]+") String requestId,
            @NotNull @Min(0) Long expectedRevision, @Size(max = 200) String title, Boolean archived) {
    }

    public record CommentCreate(
            @NotBlank @Size(max = 128) @Pattern(regexp = "[A-Za-z0-9._:-]+") String requestId,
            @NotBlank String id, String nodeId, String parentId,
            @NotBlank @Size(max = 16000) String content) {
    }

    public record CommentPatch(
            @NotBlank @Size(max = 128) @Pattern(regexp = "[A-Za-z0-9._:-]+") String requestId,
            @NotNull @Min(0) Long expectedVersion, @Size(max = 16000) String content, Boolean resolved,
            Boolean deleted) {
    }

    public record CheckpointCreate(
            @NotBlank @Size(max = 128) @Pattern(regexp = "[A-Za-z0-9._:-]+") String requestId,
            @NotBlank String id, @NotBlank @Size(max = 200) String title, @NotNull @Min(0) Long expectedRevision) {
    }
}
