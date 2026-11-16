package com.stonewu.fusion.service.resource;

import com.stonewu.fusion.repository.resource.ScriptMutationRepository;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.model.ResourceCommand;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.resource.model.ResourceCommandReceipt;
import com.stonewu.fusion.service.resource.model.ResourceMutation;
import com.stonewu.fusion.service.resource.model.ResourceOperation;
import com.stonewu.fusion.service.resource.model.ScriptClearCommand;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.SCRIPT;
import static com.stonewu.fusion.service.canvas.model.CanvasValueType.SCRIPT_EPISODE;
import static com.stonewu.fusion.service.canvas.model.CanvasValueType.SCRIPT_SCENE;

/** 清空剧本内容，保留项目固定容器及已有分镜、资产。 */
@Component
@RequiredArgsConstructor
public class ScriptClearHandler implements ResourceCommandHandler {
    private final ScriptCommandSupport support;
    private final ScriptMutationRepository writes;
    private final ScriptFields fields;
    private final CanvasJson json;

    @Override
    public CanvasValueType resourceType() {
        return SCRIPT;
    }

    @Override
    public ResourceOperation operation() {
        return ResourceOperation.CLEAR_SCRIPT_CONTENT;
    }

    @Override
    public void evict() {
        support.evict();
    }

    @Override
    public Long authorize(ResourceCommand input, ResourceCommandContext context, ResourceCommandReceipt previous) {
        return support.authorize(SCRIPT, ((ScriptClearCommand) input).resourceId(), context, false);
    }

    @Override
    public ResourceMutation apply(ResourceCommand input, ResourceCommandContext context) {
        var command = (ScriptClearCommand) input;
        var change = writes.lock(SCRIPT, command.resourceId());
        ResourceMutations.requireVersion(change.target, command.expectedVersion());
        for (var episode : change.children(SCRIPT_EPISODE, command.resourceId())) {
            change.children(SCRIPT_SCENE, episode.resourceId()).forEach(change::delete);
            change.delete(episode);
        }
        var draft = json.object(Map.of("parsingStatus", 0, "status", 0, "sourceType", 1));
        for (String field : List.of("rawContent", "content", "storySynopsis", "charactersJson", "parsingProgress",
                "summary", "genre", "targetAudience", "durationEstimate")) {
            draft.putNull(field);
        }
        var values = fields.values(SCRIPT, draft);
        values.put("totalEpisodes", 0);
        return change.result(change.patch(change.script, values));
    }
}
