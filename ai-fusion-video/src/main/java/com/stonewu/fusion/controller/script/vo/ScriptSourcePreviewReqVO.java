package com.stonewu.fusion.controller.script.vo;

import com.stonewu.fusion.service.script.ScriptSourceSplitter.Rule;
import jakarta.validation.constraints.*;
import static com.stonewu.fusion.service.script.ScriptSourceSplitter.MAX_SOURCE_LENGTH;
import static com.stonewu.fusion.service.script.ScriptSourceSplitter.MAX_EPISODE_LENGTH;

public record ScriptSourcePreviewReqVO(@NotBlank @Size(max = MAX_SOURCE_LENGTH) String rawContent,
                                      @NotNull Rule rule, @Min(1000) @Max(MAX_EPISODE_LENGTH) int targetLength) {
}
