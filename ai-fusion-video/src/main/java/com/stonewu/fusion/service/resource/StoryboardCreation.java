package com.stonewu.fusion.service.resource;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.repository.resource.StoryboardMutationRepository.ChangeSet;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.model.ResourceHeader;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

@Component
@RequiredArgsConstructor
public class StoryboardCreation {
    private final StoryboardFields fields;
    private final StoryboardRelations relations;

    public long insert(ChangeSet change, ResourceHeader parent, ObjectNode input) {
        var existing = change.children(parent);
        ObjectNode draft = input.deepCopy();
        int index = draft.has("sortOrder") ? StoryboardFields.integer(draft.get("sortOrder"), 0, existing.size(),
                "sortOrder") : existing.size();
        if (draft.has("insertAfterId")) {
            if (draft.has("sortOrder")) {
                throw StoryboardFields.invalid("不能同时指定两种插入位置");
            }
            long after = StoryboardFields.id(draft.remove("insertAfterId"), "insertAfterId");
            index = -1;
            for (int position = 0; position < existing.size(); position++) {
                if (existing.get(position).resourceId() == after) {
                    index = position + 1;
                }
            }
            if (index < 0) {
                throw StoryboardFields.invalid("插入位置不属于当前集合");
            }
        }
        long id = createAt(change, parent, draft, index);
        List<Long> order = new ArrayList<>(existing.stream().map(ResourceHeader::resourceId).toList());
        order.add(index, id);
        change.reorder(parent, order);
        return id;
    }

    /**
     * 数组顺序是业务顺序；嵌套创建受同一个 ChangeSet 的总数上限保护。
     */
    public long createAt(ChangeSet change, ResourceHeader parent, ObjectNode input, int index) {
        CanvasValueType type = StoryboardFields.child(parent.resourceType());
        ObjectNode draft = input.deepCopy();
        JsonNode nested = null;
        if (type == STORYBOARD_EPISODE) {
            nested = draft.remove("scenes");
        }
        if (type == STORYBOARD_SCENE) {
            nested = draft.remove("shots");
        }
        if (nested != null && (!nested.isArray() || nested.size() > 500)) {
            throw StoryboardFields.invalid("嵌套分镜结构必须为最多 500 项的数组");
        }
        Map<String, Object> values = fields.values(type, draft, true);
        if (values.containsKey("sortOrder") && (Integer) values.get("sortOrder") != index) {
            throw StoryboardFields.invalid("条目 sortOrder 必须与本次集合顺序一致");
        }
        relations.validate(change.header(STORYBOARD, change.board.resourceId()), type, draft, null);
        values.put("storyboardId", change.board.resourceId());
        values.put("sortOrder", index);
        if (type == STORYBOARD_SCENE) {
            values.put("episodeId", parent.resourceId());
            values.putIfAbsent("sceneNumber", change.data(parent).path("episodeNumber").intValue() + "-" + (index + 1));
        } else if (type == SHOT) {
            values.put("storyboardEpisodeId", change.data(parent).path("episodeId").longValue());
            values.put("storyboardSceneId", parent.resourceId());
            values.putIfAbsent("shotNumber", String.valueOf(index + 1));
            values.put("autoShotNumber", String.valueOf(index + 1));
        }
        ObjectNode result = change.create(type, values);
        long id = result.path("id").longValue();
        if (nested != null) {
            var node = change.header(type, id);
            int next = 0;
            for (JsonNode child : nested) {
                if (!(child instanceof ObjectNode object)) {
                    throw StoryboardFields.invalid("嵌套条目必须为对象");
                }
                createAt(change, node, object, next++);
            }
        }
        return id;
    }
}
