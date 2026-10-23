package com.stonewu.fusion.repository.resource;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.ProjectFields;
import com.stonewu.fusion.service.resource.ResourceMutations;
import com.stonewu.fusion.service.resource.model.ResourceChange;
import com.stonewu.fusion.service.resource.model.ResourceHeader;
import com.stonewu.fusion.service.resource.model.ResourceMutation;
import lombok.RequiredArgsConstructor;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.stonewu.fusion.mapper.resource.ResourceMutationMapper;
import com.stonewu.fusion.mapper.resource.ResourceMutationMapper.NewResource;
import com.stonewu.fusion.mapper.resource.ResourceReadMapper;
import com.stonewu.fusion.mapper.project.ProjectMemberMapper;
import com.stonewu.fusion.mapper.script.ScriptMapper;
import com.stonewu.fusion.mapper.script.ScriptEpisodeMapper;
import com.stonewu.fusion.mapper.script.ScriptSceneItemMapper;
import com.stonewu.fusion.mapper.storyboard.StoryboardMapper;
import com.stonewu.fusion.mapper.storyboard.StoryboardEpisodeMapper;
import com.stonewu.fusion.mapper.storyboard.StoryboardSceneMapper;
import com.stonewu.fusion.mapper.storyboard.StoryboardItemMapper;
import com.stonewu.fusion.entity.script.ScriptEpisode;
import com.stonewu.fusion.entity.script.ScriptSceneItem;
import com.stonewu.fusion.entity.storyboard.Storyboard;
import com.stonewu.fusion.entity.storyboard.StoryboardEpisode;
import com.stonewu.fusion.entity.storyboard.StoryboardScene;
import com.stonewu.fusion.entity.storyboard.StoryboardItem;

import org.springframework.stereotype.Repository;

import java.util.*;

@Repository
@RequiredArgsConstructor
public class ProjectWorkspaceRepository {
    private static final List<CanvasValueType> CHILD_TYPES = List.of(CanvasValueType.ASSET, CanvasValueType.ASSET_ITEM,
            CanvasValueType.SCRIPT, CanvasValueType.SCRIPT_EPISODE, CanvasValueType.SCRIPT_SCENE,
            CanvasValueType.STORYBOARD,
            CanvasValueType.STORYBOARD_EPISODE, CanvasValueType.STORYBOARD_SCENE, CanvasValueType.SHOT);
    private final ResourceMutationMapper mutations;
    private final ResourceReadMapper records;
    private final ProjectMemberMapper members;
    private final ScriptMapper scripts;
    private final ScriptEpisodeMapper episodes;
    private final ScriptSceneItemMapper scenes;
    private final StoryboardMapper boards;
    private final StoryboardEpisodeMapper boardEpisodes;
    private final StoryboardSceneMapper boardScenes;
    private final StoryboardItemMapper shots;
    private final ResourceReadRepository resources;
    private final ProjectFields fields;
    private final CanvasJson json;

    public record Workspace(ObjectNode result, List<ResourceChange> changes) {
    }

    public ResourceMutation create(ObjectNode values, CanvasActor actor) {
        Map<String, Object> validated = fields.values(values, true);
        Map<String, Object> columns = new LinkedHashMap<>();
        validated.forEach((key, value) -> columns.put(fields.column(key), value));
        columns.put("owner_type", actor.teamId() == null ? 1 : 2);
        columns.put("owner_id", actor.teamId() == null ? actor.userId() : actor.teamId());
        columns.put("scope", 2);
        columns.put("version", 0);
        var row = new NewResource(CanvasValueType.PROJECT, columns);
        mutations.insertCopy(row);
        long id = row.getId();
        members.createOwner(id, actor.userId());
        ResourceHeader header = resources.require(CanvasValueType.PROJECT, id, true);
        Workspace workspace = ensure(header, false);
        List<ResourceChange> changes = new ArrayList<>();
        changes.add(ResourceMutations.change(CanvasValueType.PROJECT, id, id, 0, false, List.of("*")));
        changes.addAll(workspace.changes());
        return new ResourceMutation(resources.data(header), changes, json.object(Map.of("created", changes)));
    }

    /**
     * 调用方已锁住项目；读取有效容器，缺失时显式初始化，始终保留已存在内容。
     */
    public Workspace ensure(ResourceHeader project, boolean updateProjectVersion) {
        ObjectNode data = resources.data(project);
        List<ResourceChange> changes = new ArrayList<>();
        Long scriptId = scripts.lockLatest(project.resourceId());
        if (scriptId == null) {
            scriptId = createContainer(CanvasValueType.SCRIPT, project, data, null);
            changes.add(ResourceMutations.change(CanvasValueType.SCRIPT, scriptId, project.resourceId(), 0, false,
                    List.of("*")));
        }
        Long storyboardId = boards.lockLatest(project.resourceId(), scriptId);
        if (storyboardId == null) {
            storyboardId = createContainer(CanvasValueType.STORYBOARD, project, data, scriptId);
            changes.add(
                    ResourceMutations.change(CanvasValueType.STORYBOARD, storyboardId, project.resourceId(), 0, false,
                            List.of("*")));
        }
        if (!changes.isEmpty() && updateProjectVersion) {
            ResourceMutations.requireVersion(project, project.version());
            requireUpdated(
                    mutations.patch(CanvasValueType.PROJECT, project.resourceId(), project.version(), Map.of(), false));
            changes.add(ResourceMutations.change(CanvasValueType.PROJECT, project.resourceId(), project.resourceId(),
                    project.version() + 1, false, List.of("workspace")));
        }
        return new Workspace(overview(project.resourceId(), scriptId, storyboardId), List.copyOf(changes));
    }

    private long createContainer(CanvasValueType type, ResourceHeader project, ObjectNode data, Long scriptId) {
        Map<String, Object> columns = new LinkedHashMap<>();
        columns.put("project_id", project.resourceId());
        columns.put("title", data.path("name").textValue());
        columns.put("scope", data.path("scope").numberValue());
        columns.put("owner_type", data.path("ownerType").intValue());
        columns.put("owner_id", data.path("ownerId").longValue());
        columns.put("version", 0);
        if (type == CanvasValueType.STORYBOARD) {
            columns.put("script_id", scriptId);
        }
        var row = new NewResource(type, columns);
        mutations.insertCopy(row);
        return row.getId();
    }

    public ObjectNode overview(long projectId) {
        Long scriptId = resources.scriptForProject(projectId).orElse(null);
        Long storyboardId = scriptId == null ? null : boards.selectPage(new Page<Storyboard>(1, 1, false),
                        new LambdaQueryWrapper<Storyboard>().select(Storyboard::getId).eq(Storyboard::getProjectId, projectId)
                                .eq(Storyboard::getScriptId, scriptId)
                                .orderByDesc(Storyboard::getCreateTime, Storyboard::getId))
                .getRecords().stream().map(Storyboard::getId).findFirst().orElse(null);
        return overview(projectId, scriptId, storyboardId);
    }

    private ObjectNode overview(long projectId, Long scriptId, Long storyboardId) {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("projectVersion", resources.require(CanvasValueType.PROJECT, projectId, false).version());
        result.put("script",
                scriptId == null ? null : resources.data(resources.require(CanvasValueType.SCRIPT, scriptId, false)));
        result.put("scriptEpisodeCount", scriptId == null ? 0 : episodes.selectCount(
                new LambdaQueryWrapper<ScriptEpisode>().eq(ScriptEpisode::getScriptId, scriptId)));
        result.put("scriptSceneCount", scriptId == null ? 0 : scenes.selectCount(
                new LambdaQueryWrapper<ScriptSceneItem>().eq(ScriptSceneItem::getScriptId, scriptId)));
        result.put("storyboard", storyboardId == null ? null : resources.data(
                resources.require(CanvasValueType.STORYBOARD, storyboardId, false)));
        result.put("storyboardStatistics", Map.of("episodeCount",
                storyboardId == null ? 0 : boardEpisodes.selectCount(
                        new LambdaQueryWrapper<StoryboardEpisode>().eq(StoryboardEpisode::getStoryboardId,
                                storyboardId)),
                "sceneCount", storyboardId == null ? 0 : boardScenes.selectCount(
                        new LambdaQueryWrapper<StoryboardScene>().eq(StoryboardScene::getStoryboardId, storyboardId)),
                "itemCount", storyboardId == null ? 0 : shots.selectCount(
                        new LambdaQueryWrapper<StoryboardItem>().eq(StoryboardItem::getStoryboardId, storyboardId))));
        return json.read(json.write(result), ObjectNode.class);
    }

    public ResourceMutation delete(ResourceHeader project) {
        List<Map<String, Object>> previousMembers = members.lockProjectMembers(project.resourceId());
        List<ResourceHeader> affected = new ArrayList<>();
        for (CanvasValueType type : CHILD_TYPES) {
            affected.addAll(records.lockProjectResources(type, project.resourceId()).stream()
                    .map(row -> new ResourceHeader(type, row.id(), project.resourceId(), row.version(), false))
                    .toList());
        }
        affected.add(project);
        List<ObjectNode> previous = new ArrayList<>();
        for (ResourceHeader header : affected) {
            ResourceMutations.requireVersion(header, header.version());
            previous.add(json.object(Map.of("header", header, "data", resources.data(header))));
        }
        List<ResourceChange> changes = new ArrayList<>();
        for (ResourceHeader header : affected) {
            requireUpdated(mutations.patch(header.resourceType(), header.resourceId(), header.version(),
                    Map.of("deleted", true), false));
            changes.add(ResourceMutations.change(header.resourceType(), header.resourceId(), project.resourceId(),
                    header.version() + 1, true, List.of("deleted")));
        }
        members.removeProjectMembers(project.resourceId());
        return new ResourceMutation(json.object(Map.of("id", project.resourceId(), "version", project.version() + 1,
                "deleted", true, "affectedCount", changes.size())), changes,
                json.object(Map.of("resources", previous, "members", previousMembers)));
    }

    private void requireUpdated(int count) {
        if (count != 1) {
            throw new CanvasException(409, CanvasErrorCode.VERSION_CONFLICT, "项目内容已变化", Map.of());
        }
    }
}
