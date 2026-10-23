package com.stonewu.fusion.repository.resource;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.stonewu.fusion.entity.project.Project;
import com.stonewu.fusion.entity.script.Script;
import com.stonewu.fusion.entity.script.ScriptEpisode;
import com.stonewu.fusion.entity.script.ScriptSceneItem;
import com.stonewu.fusion.mapper.project.ProjectMapper;
import com.stonewu.fusion.mapper.script.ScriptMapper;
import com.stonewu.fusion.mapper.script.ScriptEpisodeMapper;
import com.stonewu.fusion.mapper.script.ScriptSceneItemMapper;
import com.stonewu.fusion.mapper.resource.ResourceReadMapper;

import java.time.LocalDateTime;
import java.time.ZoneId;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.model.ResourceHeader;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Repository;

import java.time.temporal.TemporalAccessor;
import java.util.*;

@Repository
@RequiredArgsConstructor
public class ResourceReadRepository {
    // 业务 JSON 字段及历史 TEXT 类型的 prop_ids 统一保持结构化快照协议。
    private static final Set<String> JSON_COLUMNS = Set.of("properties", "tags", "characters_json", "characters",
            "character_asset_ids", "prop_asset_ids", "dialogues", "custom_columns", "character_ids", "prop_ids",
            "custom_data");
    private final ResourceReadMapper records;
    private final ProjectMapper projects;
    private final ScriptMapper scripts;
    private final ScriptEpisodeMapper episodes;
    private final ScriptSceneItemMapper scenes;
    private final CanvasJson json;

    public Optional<ResourceHeader> find(CanvasValueType type, long id, boolean lock, boolean includeDeleted) {
        ResourceTables.of(type);
        return Optional.ofNullable(records.header(type, id, false, includeDeleted, lock))
                .map(row -> new ResourceHeader(type, row.id(), row.projectId(), row.version(), row.deleted()));
    }

    public ResourceHeader require(CanvasValueType type, long id, boolean lock) {
        return find(type, id, lock, false).orElseThrow(() -> new CanvasException(404, CanvasErrorCode.RESOURCE_DELETED,
                "业务对象已删除或不存在", Map.of("resourceType", type.code(), "resourceId", id)));
    }

    public ResourceHeader historical(CanvasValueType type, long id) {
        ResourceTables.of(type);
        return Optional.ofNullable(records.header(type, id, true, true, false))
                .map(row -> new ResourceHeader(type, row.id(), row.projectId(), row.version(), row.deleted()))
                .orElseThrow(() -> new CanvasException(404, CanvasErrorCode.RESOURCE_DELETED,
                        "历史对象的授权记录不存在", Map.of()));
    }

    public ObjectNode data(ResourceHeader header) {
        ResourceTables.of(header.resourceType());
        return Optional.ofNullable(records.data(header.resourceType(), header.resourceId(), header.version()))
                .map(this::data).orElseThrow(() -> new CanvasException(409, CanvasErrorCode.VERSION_CONFLICT,
                        "业务内容与读取版本不一致", Map.of()));
    }

    /**
     * 仅供同一业务事务的审计/恢复使用；调用方必须完成当前对象的授权。
     */
    public ObjectNode raw(CanvasValueType type, long id, boolean lock) {
        ResourceTables.of(type);
        return Optional.ofNullable(records.raw(type, id, lock)).map(this::data)
                .orElseThrow(() -> new CanvasException(404, CanvasErrorCode.RESOURCE_DELETED,
                        "业务对象记录不存在", Map.of()));
    }

    public List<ObjectNode> projects(List<Long> ids) {
        if (ids.isEmpty()) {
            return List.of();
        }
        return projects.selectMaps(new LambdaQueryWrapper<Project>().in(Project::getId, ids)
                .orderByDesc(Project::getUpdateTime, Project::getId)).stream().map(this::data).toList();
    }

    public Optional<Long> scriptForProject(long projectId) {
        return scripts.selectPage(new Page<Script>(1, 1, false), new LambdaQueryWrapper<Script>()
                        .select(Script::getId).eq(Script::getProjectId, projectId)
                        .orderByDesc(Script::getCreateTime, Script::getId)).getRecords().stream().map(Script::getId)
                .findFirst();
    }

    public List<Long> scriptChildren(CanvasValueType type, long parentId) {
        return switch (type) {
            case SCRIPT_EPISODE -> episodes.selectList(new LambdaQueryWrapper<ScriptEpisode>()
                            .select(ScriptEpisode::getId).eq(ScriptEpisode::getScriptId, parentId)
                            .orderByAsc(ScriptEpisode::getSortOrder, ScriptEpisode::getId))
                    .stream().map(ScriptEpisode::getId).toList();
            case SCRIPT_SCENE -> scenes.selectList(new LambdaQueryWrapper<ScriptSceneItem>()
                            .select(ScriptSceneItem::getId).eq(ScriptSceneItem::getEpisodeId, parentId)
                            .orderByAsc(ScriptSceneItem::getSortOrder, ScriptSceneItem::getId))
                    .stream().map(ScriptSceneItem::getId).toList();
            default -> throw new CanvasException(CanvasErrorCode.INVALID_RESOURCE_REFERENCE, "不是剧本子对象");
        };
    }

    public List<ObjectNode> projectPage(List<Long> ids, int page, int size) {
        if (ids.isEmpty()) {
            return List.of();
        }
        return projects.selectMapsPage(new Page<Map<String, Object>>(page, size, false),
                        new LambdaQueryWrapper<Project>().in(Project::getId, ids)
                                .orderByDesc(Project::getUpdateTime, Project::getId))
                .getRecords().stream().map(this::data).toList();
    }

    private ObjectNode data(Map<String, Object> row) {
        Map<String, Object> values = new LinkedHashMap<>();
        for (var entry : row.entrySet()) {
            String column = entry.getKey();
            Object value = entry.getValue();
            if (value != null && JSON_COLUMNS.contains(column)) {
                value = json.read(value.toString(), JsonNode.class);
            } else if (value instanceof Date date) {
                value = LocalDateTime.ofInstant(date.toInstant(), ZoneId.systemDefault()).toString();
            } else if (value instanceof TemporalAccessor time) {
                value = time.toString();
            }
            if (("deleted".equals(column) || "ai_generated".equals(column)) && value instanceof Number number) {
                if (number.intValue() != 0 && number.intValue() != 1) {
                    throw new IllegalStateException("业务布尔字段无效");
                }
                value = number.intValue() == 1;
            }
            values.put(camel(column), value);
        }
        return json.object(values);
    }

    private String camel(String column) {
        StringBuilder result = new StringBuilder();
        boolean upper = false;
        for (char character : column.toCharArray()) {
            if (character == '_') {
                upper = true;
            } else {
                result.append(upper ? Character.toUpperCase(character) : character);
                upper = false;
            }
        }
        return result.toString();
    }
}
