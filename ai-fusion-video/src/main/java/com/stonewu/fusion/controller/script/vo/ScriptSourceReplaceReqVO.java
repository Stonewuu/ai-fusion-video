package com.stonewu.fusion.controller.script.vo;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.PositiveOrZero;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.Size;
import com.stonewu.fusion.service.resource.model.ScriptSourceCommand.Mode;
import com.stonewu.fusion.service.script.ScriptSourceSplitter.Segment;
import java.util.List;
import static com.stonewu.fusion.service.script.ScriptSourceSplitter.MAX_SOURCE_LENGTH;
import static com.stonewu.fusion.service.script.ScriptSourceSplitter.MAX_EPISODES;

public record ScriptSourceReplaceReqVO(@NotBlank String requestId, @NotNull @PositiveOrZero Integer version,
                                       @NotBlank @Size(max = MAX_SOURCE_LENGTH) String rawContent,
                                       @NotNull Mode mode,
                                       @NotEmpty @Size(max = MAX_EPISODES) List<@Valid Segment> segments) {
}
