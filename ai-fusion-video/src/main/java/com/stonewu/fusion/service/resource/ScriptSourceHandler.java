package com.stonewu.fusion.service.resource;

import com.stonewu.fusion.repository.resource.ScriptMutationRepository;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

@Component
@RequiredArgsConstructor
public class ScriptSourceHandler implements ResourceCommandHandler {
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
        return ResourceOperation.REPLACE_SCRIPT_SOURCE;
    }

    @Override
    public void evict() {
        support.evict();
    }

    @Override
    public Long authorize(ResourceCommand input, ResourceCommandContext context, ResourceCommandReceipt previous) {
        return support.authorize(SCRIPT, ((ScriptSourceCommand) input).resourceId(), context, false);
    }

    @Override
    public ResourceMutation apply(ResourceCommand input, ResourceCommandContext context) {
        var command = (ScriptSourceCommand) input;
        var change = writes.lock(SCRIPT, command.resourceId());
        ResourceMutations.requireVersion(change.target, command.expectedVersion());
        var draft = json.object(Map.of("rawContent", command.rawContent(), "parsingStatus", 0));
        for (String key : List.of("content", "storySynopsis", "charactersJson", "parsingProgress", "summary", "genre",
                "targetAudience", "durationEstimate")) {
            draft.putNull(key);
        }
        var values = fields.values(SCRIPT, draft);
        values.put("totalEpisodes", 0);
        for (var episode : change.children(SCRIPT_EPISODE, command.resourceId())) {
            change.children(SCRIPT_SCENE, episode.resourceId()).forEach(change::delete);
            change.delete(episode);
        }
        return change.result(change.patch(change.script, values));
    }
}
