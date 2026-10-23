package com.stonewu.fusion.service.resource;

import com.stonewu.fusion.repository.resource.ProjectForkRepository;
import com.stonewu.fusion.repository.resource.ProjectWorkspaceRepository;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.collaboration.CanvasProjectForkSources.Entry;
import com.stonewu.fusion.service.canvas.collaboration.CanvasProjectForkValues;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.media.MediaCopyService;
import com.stonewu.fusion.service.media.model.MediaResource;
import com.stonewu.fusion.service.resource.ResourceForkReferences.Key;
import com.stonewu.fusion.service.resource.model.ResourceChange;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 已授权本地来源或已验证模板都创建新的业务归属，不更新来源对象。
 */
@Component
@RequiredArgsConstructor
public class ProjectContentCopies {
    private final ProjectWorkspaceRepository workspaces;
    private final ProjectForkRepository repository;
    private final MediaCopyService mediaCopies;
    private final CanvasJson json;

    public record Copied(long projectId, CanvasProjectForkValues remap, List<ResourceChange> changes,
                         Map<Key, Long> objects, Map<String, MediaResource> media) {
    }

    @Transactional(propagation = Propagation.MANDATORY)
    public Copied create(String title, String operation, Long sourceProject, Long primaryScript, Long primaryStoryboard,
                         List<Entry> entries, Map<String, MediaResource> files, CanvasActor actor) {
        var created = workspaces.create(json.object(Map.of("name", title)), actor);
        long project = created.result().path("id").longValue();
        var workspace = workspaces.overview(project);
        Map<Key, Long> ids = new LinkedHashMap<>();
        if (sourceProject != null) {
            ids.put(new Key(CanvasValueType.PROJECT, sourceProject), project);
        }
        if (primaryScript != null) {
            ids.put(new Key(CanvasValueType.SCRIPT, primaryScript), workspace.path("script").path("id").longValue());
        }
        if (primaryStoryboard != null) {
            ids.put(new Key(CanvasValueType.STORYBOARD, primaryStoryboard),
                    workspace.path("storyboard").path("id").longValue());
        }
        Map<String, MediaResource> media = new LinkedHashMap<>();
        files.entrySet().stream().sorted(Comparator.comparing(entry -> entry.getValue().id())).forEach(entry -> {
            var file = entry.getValue();
            media.put(entry.getKey(), mediaCopies.copy(file.id(), file.version(), project, operation, actor));
        });
        var remap = new CanvasProjectForkValues(project, actor, ids, media, json);
        Map<Key, ResourceChange> changes = new LinkedHashMap<>();
        created.changes().forEach(change -> changes.put(new Key(change.resourceType(), change.resourceId()), change));
        for (var entry : entries.stream()
                .sorted(Comparator.comparingInt((Entry value) -> ProjectForkRepository.ORDER.indexOf(value.type()))
                        .thenComparingLong(Entry::id)).toList()) {
            var data = remap.business(entry.type(), entry.data());
            if (List.of(CanvasValueType.PROJECT, CanvasValueType.ASSET, CanvasValueType.SCRIPT,
                    CanvasValueType.STORYBOARD).contains(entry.type())) {
                data.put("ownerType", actor.teamId() == null ? 1 : 2);
                data.put("ownerId", actor.teamId() == null ? actor.userId() : actor.teamId());
            }
            if (entry.type() == CanvasValueType.ASSET) {
                data.put("userId", actor.userId());
            }
            if (List.of(CanvasValueType.PROJECT, CanvasValueType.SCRIPT, CanvasValueType.STORYBOARD)
                    .contains(entry.type())) {
                data.put("scope", 2);
            }
            int version = Math.addExact(entry.data().path("version").intValue(), 1);
            data.put("version", version);
            if (entry.type() == CanvasValueType.PROJECT) {
                data.put("name", title);
                data.put("status", 0);
            }
            if (entry.type() == CanvasValueType.ASSET) {
                data.put("projectId", project);
            }
            if (data.path("parsingStatus").asInt() == 1) {
                data.put("parsingStatus", 0);
                if (data.has("parsingProgress")) {
                    data.putNull("parsingProgress");
                }
            }
            if (data.path("composeStatus").asInt() == 1) {
                data.put("composeStatus", 0);
                data.putNull("composeErrorMsg");
            }
            var key = new Key(entry.type(), entry.id());
            long id = repository.write(entry.type(), ids.get(key), data);
            ids.put(key, id);
            changes.put(new Key(entry.type(), id),
                    ResourceMutations.change(entry.type(), id, project, version, data.path("deleted").asBoolean(),
                            List.of("*")));
        }
        return new Copied(project, remap, List.copyOf(changes.values()), Map.copyOf(ids), Map.copyOf(media));
    }
}
