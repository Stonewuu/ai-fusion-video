package com.stonewu.fusion.service.media.deletion;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.PositiveOrZero;

import java.util.List;

public final class MediaDeletionData {
    private MediaDeletionData() {
    }

    public enum Mode {
        LIBRARY, PERMANENT
    }

    public record Request(@NotBlank @Pattern(regexp = "[A-Za-z0-9._:-]{1,96}") String requestId,
                          @NotNull @PositiveOrZero Integer expectedVersion,
                          @NotNull Mode mode,
                          @NotBlank @Pattern(regexp = "[0-9a-f]{64}") String referenceToken) {
    }

    public record Reference(String kind, String id, String canvasId, String retention) {
    }

    public record Usage(String label, long count) {
    }

    public record Preview(String mediaId, int version, String referenceToken, List<Usage> usages,
                          long sharedCopies, String permanentBlockedReason, boolean libraryRemoved) {
    }

    public record Result(String mediaId, Mode mode, String fileStatus) {
    }
}
