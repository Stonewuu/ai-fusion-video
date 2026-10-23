package com.stonewu.fusion.service.resource.model;

import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import jakarta.validation.constraints.*;

/**
 * 已有媒体的显式采用；不伪造一次分集合成任务，也不覆盖运行中的合成。
 */
public record AdoptEpisodeMediaCommand(
        @NotBlank @Size(max = 128) @Pattern(regexp = "[A-Za-z0-9._:-]+") String requestId,
        @Positive long episodeId, @NotNull @PositiveOrZero Integer expectedVersion,
        @NotBlank String field, @NotBlank String mediaId) implements ResourceCommand {
    @Override
    public CanvasValueType resourceType() {
        return CanvasValueType.STORYBOARD_EPISODE;
    }

    @Override
    public ResourceOperation operation() {
        return ResourceOperation.ADOPT_EPISODE_MEDIA;
    }
}
