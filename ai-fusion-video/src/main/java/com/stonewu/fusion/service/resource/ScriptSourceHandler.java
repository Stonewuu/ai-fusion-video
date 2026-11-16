package com.stonewu.fusion.service.resource;

import com.stonewu.fusion.repository.resource.ScriptMutationRepository;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.model.*;
import com.stonewu.fusion.service.script.ScriptSourceSplitter;
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
    private final ScriptSourceSplitter splitter;

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
        splitter.validateSegments(command.rawContent(), command.segments());
        var change = writes.lock(SCRIPT, command.resourceId());
        ResourceMutations.requireVersion(change.target, command.expectedVersion());
        var existing = change.children(SCRIPT_EPISODE, command.resourceId());
        boolean replace = command.mode() == ScriptSourceCommand.Mode.REPLACE;
        int count = replace ? 0 : existing.size();
        if (count + command.segments().size() > ScriptSourceSplitter.MAX_EPISODES) {
            throw ScriptFields.invalid("一个剧本最多保存 20000 集，请减少本次导入数量");
        }
        int number = 0;
        int order = 0;
        for (var episode : existing) {
            if (replace) {
                change.children(SCRIPT_SCENE, episode.resourceId()).forEach(change::delete);
                change.delete(episode);
            } else {
                var data = change.data(episode);
                number = Math.max(number, data.path("episodeNumber").intValue());
                order = Math.max(order, data.path("sortOrder").intValue());
            }
        }
        if (number + command.segments().size() > 1_000_000 || order + command.segments().size() > 1_000_000) {
            throw ScriptFields.invalid("集号已达到上限，请调整现有集号后导入");
        }
        for (var segment : command.segments()) {
            var episode = json.object(Map.of("episodeNumber", ++number, "sortOrder", ++order,
                    "title", segment.title().strip(), "rawContent", command.rawContent().substring(segment.start(), segment.end()),
                    "sourceType", 2, "parsingStatus", 0));
            var values = fields.values(SCRIPT_EPISODE, episode);
            values.put("scriptId", command.resourceId());
            change.create(SCRIPT_EPISODE, values);
        }
        var draft = json.object(Map.of("parsingStatus", 0));
        if (replace) {
            for (String key : List.of("rawContent", "content", "storySynopsis", "charactersJson", "parsingProgress", "summary", "genre",
                    "targetAudience", "durationEstimate")) {
                draft.putNull(key);
            }
        }
        var values = fields.values(SCRIPT, draft);
        values.put("totalEpisodes", count + command.segments().size());
        var result = change.patch(change.script, values);
        result.remove(List.of("rawContent", "content"));
        return change.result(result);
    }
}
