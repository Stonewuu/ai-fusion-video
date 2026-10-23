package com.stonewu.fusion.repository.resource;

import com.baomidou.mybatisplus.core.toolkit.Wrappers;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.model.ResourceChange;
import com.stonewu.fusion.service.resource.model.ResourceCommandReceipt;
import com.stonewu.fusion.service.resource.model.ResourceHeader;
import com.stonewu.fusion.service.resource.model.ResourceMutation;
import lombok.RequiredArgsConstructor;
import com.stonewu.fusion.mapper.resource.ResourceRevisionMapper;
import com.stonewu.fusion.entity.resource.ResourceRevisionEntity;
import org.springframework.stereotype.Repository;

import java.util.*;

@Repository
@RequiredArgsConstructor
public class ResourceRevisionRepository {
    private final ResourceRevisionMapper revisions;
    private final CanvasJson json;
    private final ResourceReadRepository resources;
    private static final Set<String> METADATA = Set.of("version", "createTime", "updateTime");

    public record Revision(String commandId, CanvasValueType resourceType, long resourceId, Long projectId, int version,
                           ObjectNode before, ObjectNode after, List<String> fields) {
    }

    public void capture(ResourceCommandReceipt receipt, ResourceMutation mutation) {
        Map<String, ObjectNode> previous = new LinkedHashMap<>();
        collect(mutation.previousValues(), previous);
        Set<String> created = new LinkedHashSet<>();
        collectCreated(mutation.previousValues(), created);
        Map<String, ResourceChange> first = new LinkedHashMap<>(), latest = new LinkedHashMap<>();
        for (var change : mutation.changes()) {
            String key = key(change.resourceType(), change.resourceId());
            first.putIfAbsent(key, change);
            latest.put(key, change);
        }
        for (var item : latest.entrySet()) {
            var change = item.getValue();
            boolean creation = first.get(item.getKey()).version() == 0 || created.contains(item.getKey());
            ObjectNode before = creation ? null : previous.get(item.getKey());
            if (before == null && !creation) {
                throw new IllegalStateException("业务命令缺少变更前快照：" + item.getKey());
            }
            ObjectNode after = resources.raw(change.resourceType(), change.resourceId(), false);
            if (after.path("version").intValue() != change.version()) {
                throw new IllegalStateException("业务结果与提交版本不一致");
            }
            var fields = changedFields(before, after);
            revisions.capture(ResourceRevisionEntity.builder().commandId(receipt.commandId())
                    .resourceType(change.resourceType().code()).resourceId(change.resourceId())
                    .projectId(change.projectId())
                    .resourceVersion(change.version()).beforeJson(before == null ? null : json.write(before))
                    .afterJson(json.write(after)).changedFields(json.write(fields)).build());
        }
    }

    public List<Revision> command(String commandId) {
        return revisions.selectList(Wrappers.<ResourceRevisionEntity>lambdaQuery()
                        .eq(ResourceRevisionEntity::getCommandId, commandId)
                        .orderByAsc(ResourceRevisionEntity::getResourceType, ResourceRevisionEntity::getResourceId)).stream()
                .map(row -> new Revision(row.getCommandId(),
                        CanvasValueType.fromCode(row.getResourceType()), row.getResourceId(), row.getProjectId(),
                        row.getResourceVersion(),
                        row.getBeforeJson() == null ? null : json.read(row.getBeforeJson(), ObjectNode.class),
                        json.read(row.getAfterJson(), ObjectNode.class),
                        List.of(json.read(row.getChangedFields(), String[].class)))).toList();
    }

    private void collect(ObjectNode values, Map<String, ObjectNode> result) {
        for (var entry : values.path("resources")) {
            var header = json.read(json.write(entry.path("header")), ResourceHeader.class);
            result.putIfAbsent(key(header.resourceType(), header.resourceId()),
                    ((ObjectNode) entry.path("data")).deepCopy());
        }
        for (var operation : values.path("operations")) {
            collect((ObjectNode) operation, result);
        }
    }

    private void collectCreated(ObjectNode values, Set<String> result) {
        for (var entry : values.path("created")) {
            var change = json.read(entry.toString(), ResourceChange.class);
            result.add(key(change.resourceType(), change.resourceId()));
        }
        for (var operation : values.path("operations")) {
            collectCreated((ObjectNode) operation, result);
        }
    }

    public static List<String> changedFields(ObjectNode before, ObjectNode after) {
        Set<String> names = new LinkedHashSet<>();
        if (before != null) {
            before.fieldNames().forEachRemaining(names::add);
        }
        after.fieldNames().forEachRemaining(names::add);
        names.removeAll(METADATA);
        return names.stream().filter(field -> before == null || !before.path(field).equals(after.path(field))).toList();
    }

    private String key(CanvasValueType type, long id) {
        return type.code() + ":" + id;
    }
}
