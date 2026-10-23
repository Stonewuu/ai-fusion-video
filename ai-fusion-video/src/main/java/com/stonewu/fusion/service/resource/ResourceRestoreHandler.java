package com.stonewu.fusion.service.resource;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.repository.resource.ResourceTables;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.ResourceRestorePlanner.Change;
import com.stonewu.fusion.service.resource.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.dao.DataIntegrityViolationException;
import com.stonewu.fusion.mapper.resource.ResourceMutationMapper;
import org.springframework.stereotype.Component;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

@Component
@RequiredArgsConstructor
public class ResourceRestoreHandler implements ResourceCommandHandler {
    private final ResourceRestorePlanner planner;
    private final ResourceMutationMapper mutations;
    private final CanvasJson json;

    @Override
    public CanvasValueType resourceType() {
        return CanvasValueType.PROJECT;
    }

    @Override
    public Set<CanvasValueType> resourceTypes() {
        return Arrays.stream(CanvasValueType.values()).filter(CanvasValueType::isBusinessResource)
                .collect(Collectors.toSet());
    }

    @Override
    public ResourceOperation operation() {
        return ResourceOperation.RESTORE;
    }

    @Override
    public Long authorize(ResourceCommand input, ResourceCommandContext context, ResourceCommandReceipt previous) {
        if (context.canvasId() == null) {
            throw new CanvasException(CanvasErrorCode.INVALID_COMMAND, "请从具体画布的业务历史恢复");
        }
        var command = (ResourceRestoreCommand) input;
        return planner.authorize(context.canvasId(), command.commandId(), context.actor()).projectId();
    }

    @Override
    public ResourceMutation apply(ResourceCommand input, ResourceCommandContext context) {
        var command = (ResourceRestoreCommand) input;
        var plan = planner.prepare(context.canvasId(), command.commandId(), context.actor(), true);
        if (plan.resourceType() != command.resourceType()) {
            throw new CanvasException(CanvasErrorCode.INVALID_COMMAND, "恢复命令的资源类型与原记录不一致");
        }
        if (!plan.canRestore() || !plan.digest().equals(command.expectedDigest())) {
            throw new CanvasException(409, CanvasErrorCode.VERSION_CONFLICT,
                    "恢复预览已变化或存在冲突，请重新检查差异", Map.of("conflicts", plan.conflicts()));
        }
        // 先移除原操作新建的子树，再恢复旧内容，避免替换结构的集号约束出现瞬时冲突。
        var ordered = plan.changes().stream()
                .sorted(Comparator.comparingInt((Change change) -> change.removesCreatedObject() ? 0 : 1)
                        .thenComparingInt(
                                change -> (change.removesCreatedObject() ? -1 : 1) * ResourceRestorePlanner.order(
                                        change.resourceType()))
                        .thenComparingLong(Change::resourceId)).toList();
        for (Change change : ordered) {
            apply(change);
        }
        var previous = new ArrayList<ObjectNode>();
        var events = plan.changes().stream().map(change -> {
            previous.add(json.object(Map.of("header",
                    new ResourceHeader(change.resourceType(), change.resourceId(), change.projectId(),
                            change.currentVersion(),
                            change.current().path("deleted").asBoolean()), "data", change.current())));
            return ResourceMutations.change(change.resourceType(), change.resourceId(), change.projectId(),
                    change.currentVersion() + 1,
                    change.proposed().path("deleted").asBoolean(), change.fields());
        }).toList();
        return new ResourceMutation(
                json.object(Map.of("restoredCommandId", command.commandId(), "affectedCount", events.size())), events,
                json.object(Map.of("resources", previous)));
    }

    private void apply(Change change) {
        String table = ResourceTables.of(change.resourceType());
        Map<String, String> columns = mutations.columns(table)
                .stream().collect(Collectors.toMap(this::camel, Function.identity()));
        var values = new LinkedHashMap<String, Object>();
        for (String field : change.fields()) {
            String column = columns.get(field);
            if (column == null || Set.of("id", "version", "create_time", "update_time").contains(column)) {
                throw new IllegalStateException("恢复字段没有合法存储列");
            }
            values.put(column, value(change.proposed().get(field)));
        }
        try {
            if (mutations.patch(change.resourceType(), change.resourceId(), change.currentVersion(), values,
                    true) != 1) {
                throw conflict("业务内容在恢复时已变化");
            }
        } catch (DataIntegrityViolationException conflict) {
            throw conflict("恢复会与现有业务编号或结构冲突，请先处理相关内容");
        }
    }

    private Object value(JsonNode value) {
        if (value == null || value.isNull()) {
            return null;
        }
        if (value.isContainerNode()) {
            return json.write(value);
        }
        if (value.isBoolean()) {
            return value.booleanValue();
        }
        if (value.isNumber()) {
            return value.numberValue();
        }
        return value.asText();
    }

    private String camel(String value) {
        var result = new StringBuilder();
        boolean uppercase = false;
        for (char character : value.toCharArray()) {
            if (character == '_') {
                uppercase = true;
            } else {
                result.append(uppercase ? Character.toUpperCase(character) : character);
                uppercase = false;
            }
        }
        return result.toString();
    }

    @Override
    @CacheEvict(value = {"project", "projectMember", "asset", "assetItem", "script", "episode", "scene", "storyboard", "storyboardEpisode", "storyboardScene", "storyboardItem", "storyboardStatistics", "resourceDocument"}, allEntries = true)
    public void evict() {
    }

    private CanvasException conflict(String message) {
        return new CanvasException(409, CanvasErrorCode.VERSION_CONFLICT, message, Map.of());
    }
}
