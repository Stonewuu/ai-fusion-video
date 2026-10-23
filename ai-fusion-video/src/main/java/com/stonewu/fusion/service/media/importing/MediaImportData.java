package com.stonewu.fusion.service.media.importing;

import com.stonewu.fusion.service.canvas.command.CanvasCommandReceipt;
import com.stonewu.fusion.service.generation.job.GenerationClientScope;
import com.stonewu.fusion.service.media.model.MediaResource;
import com.stonewu.fusion.service.media.model.MediaSourceReference;
import com.stonewu.fusion.service.media.model.MediaSourceSnapshot;
import jakarta.validation.Valid;
import jakarta.validation.constraints.*;

import java.time.Instant;
import java.util.List;

public final class MediaImportData {
    private MediaImportData() {
    }

    public static final String KIND = "SOURCE_IMPORT";
    public static final int MAX_ITEMS = 50;

    public record Point(double x, double y) {
    }

    public record RequestedSource(@NotBlank @Size(max = 200) String title,
                                  @NotNull @Valid MediaSourceReference source) {
    }

    public record Submit(@NotBlank @Size(max = 128) @Pattern(regexp = "[A-Za-z0-9._:-]+") String requestId,
                         @NotNull @Valid GenerationClientScope scope,
                         @NotEmpty @Size(max = MAX_ITEMS) List<@Valid RequestedSource> sources, @Valid Point position) {
    }

    public record Control(@NotNull @Valid GenerationClientScope scope) {
    }

    public record Retry(@NotNull @Valid GenerationClientScope scope, @PositiveOrZero long expectedRevision,
                        @NotEmpty @Size(max = MAX_ITEMS) List<Integer> indices) {
    }

    public record Place(@NotNull @Valid GenerationClientScope scope, @Valid Point position, boolean automatic) {
    }

    public record Placement(Point point, boolean automatic) {
    }

    public record Source(String title, String nodeId, MediaSourceSnapshot snapshot) {
    }

    public record Input(int schemaVersion, List<Source> sources) {
    }

    public record Item(String status, int attempt, String resourceId, String errorCode, String message,
                       String commandId) {
    }

    public record Checkpoint(int schemaVersion, List<Item> items, String lastRetryHash) {
    }

    public record ItemView(int index, String title, MediaSourceReference source, String nodeId, String status,
                           int attempt, String errorCode, String message, String commandId, MediaResource resource) {
    }

    public record View(String id, String canvasId, String requestId, String status, long revision,
                       boolean cancelRequested,
                       boolean automaticPlacement, boolean canControl, String phase, List<ItemView> items,
                       Instant createdAt, Instant updatedAt) {
    }

    public record Summary(String id, String status, long revision, boolean cancelRequested, boolean automaticPlacement,
                          String phase, int total, int succeeded, int failed, int completed, int readyToPlace,
                          Instant createdAt) {
    }

    public record Page(List<Summary> items, int page, boolean hasMore) {
    }

    public record Withdrawal(boolean accepted, View job) {
    }

    public record Placed(View job, CanvasCommandReceipt command) {
    }
}
