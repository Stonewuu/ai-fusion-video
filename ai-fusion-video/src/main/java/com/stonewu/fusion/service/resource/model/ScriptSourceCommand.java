package com.stonewu.fusion.service.resource.model;

import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import jakarta.validation.constraints.*;
import jakarta.validation.Valid;
import com.stonewu.fusion.service.script.ScriptSourceSplitter.Segment;
import com.stonewu.fusion.service.script.ScriptSourceSplitter;
import java.util.List;
import static com.stonewu.fusion.service.script.ScriptSourceSplitter.MAX_SOURCE_LENGTH;
import static com.stonewu.fusion.service.script.ScriptSourceSplitter.MAX_EPISODES;

public record ScriptSourceCommand(
        @NotBlank @Size(max = 128) @Pattern(regexp = "[A-Za-z0-9._:-]+") String requestId,
        @Positive long resourceId,
        @NotNull @PositiveOrZero Integer expectedVersion,
        @NotBlank @Size(max = MAX_SOURCE_LENGTH) String rawContent,
        @NotNull Mode mode,
        @NotEmpty @Size(max = MAX_EPISODES) List<@Valid Segment> segments) implements ResourceCommand {
    public enum Mode { APPEND, REPLACE }

    /** 画布文本写入同样先分集，避免重新产生总剧本全文。 */
    public static ScriptSourceCommand automatic(String requestId, long id, Integer version, String source) {
        var preview = new ScriptSourceSplitter().split(source, ScriptSourceSplitter.Rule.AUTO, 12000);
        return new ScriptSourceCommand(requestId, id, version, source, Mode.REPLACE, preview.segments());
    }

    @Override
    public CanvasValueType resourceType() {
        return CanvasValueType.SCRIPT;
    }

    @Override
    public ResourceOperation operation() {
        return ResourceOperation.REPLACE_SCRIPT_SOURCE;
    }
}
