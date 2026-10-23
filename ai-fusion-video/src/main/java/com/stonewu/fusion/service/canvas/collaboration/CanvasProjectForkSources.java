package com.stonewu.fusion.service.canvas.collaboration;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.repository.canvas.CanvasGraphRepository;
import com.stonewu.fusion.repository.resource.ProjectForkRepository;
import com.stonewu.fusion.repository.resource.ResourceReadRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.delivery.CanvasExportData;
import com.stonewu.fusion.service.canvas.delivery.CanvasExportMedia;
import com.stonewu.fusion.service.canvas.delivery.CanvasExportSources;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasResourceReference;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Value;
import com.stonewu.fusion.service.media.MediaAccessPolicy;
import com.stonewu.fusion.service.media.MediaResourceQueries;
import com.stonewu.fusion.service.media.model.MediaResource;
import com.stonewu.fusion.service.resource.ResourceAccessPolicy;
import com.stonewu.fusion.service.resource.ResourceForkReferences;
import com.stonewu.fusion.service.resource.ResourceForkReferences.Key;
import lombok.RequiredArgsConstructor;
import com.stonewu.fusion.mapper.system.UserMapper;
import com.stonewu.fusion.mapper.team.TeamMapper;
import org.springframework.stereotype.Component;

import java.util.*;

@Component
@RequiredArgsConstructor
public class CanvasProjectForkSources {
    private final CanvasDocumentRepository documents;
    private final CanvasGraphRepository graphs;
    private final CanvasAccessPolicy access;
    private final ResourceReadRepository records;
    private final ResourceAccessPolicy resourceAccess;
    private final CanvasExportSources exports;
    private final CanvasExportMedia exportMedia;
    private final MediaResourceQueries media;
    private final MediaAccessPolicy mediaAccess;
    private final CanvasJson json;
    private final UserMapper users;
    private final TeamMapper teams;

    public record Entry(CanvasValueType type, long id, ObjectNode data) {
    }

    public record Frozen(CanvasExportData.Manifest manifest, List<Entry> objects, List<MediaResource> media,
                         Long primaryScript, Long primaryStoryboard, List<String> warnings, CanvasActor actor) {
    }

    public record Preview(String digest, long revision, int nodes, int businessObjects, int historicalObjects,
                          int mediaFiles, long mediaBytes, List<String> warnings, Ownership ownership) {
    }

    public record Ownership(int type, long id, String name) {
    }

    public Frozen read(String canvas, long revision, CanvasActor actor, boolean lock) {
        var document = documents.require(canvas, lock);
        access.requireView(document, actor);
        if (document.revision() != revision) {
            throw new BusinessException(409, "画布已变化，请重新检查复制范围");
        }
        if (lock) {
            if (document.projectId() != null) {
                records.require(CanvasValueType.PROJECT, document.projectId(), true);
            } else {
                var roots = new LinkedHashSet<Long>();
                for (var node : graphs.load(document).nodes()) {
                    if (node.resource() != null) {
                        var ref = node.resource();
                        if (ref.type() == CanvasValueType.ASSET) {
                            roots.add(ref.id());
                        } else if (ref.type() == CanvasValueType.ASSET_ITEM) {
                            roots.add(records.raw(ref.type(), ref.id(), false).path("assetId").longValue());
                        }
                    }
                }
                roots.stream().sorted().forEach(id -> records.raw(CanvasValueType.ASSET, id, true));
            }
            // 等待父级写锁后按当前权限重新鉴权，撤权不能被旧的读快照掩盖。
            access.requireView(documents.require(canvas, false), actor);
        }
        var manifest = exports.freeze(canvas, new CanvasExportData.Options(CanvasExportData.Format.JSON,
                document.projectId() == null ? CanvasExportData.Scope.CANVAS : CanvasExportData.Scope.PROJECT,
                List.of(), revision), actor);
        var queue = new ArrayDeque<Key>();
        var allValues = new ArrayList<>(manifest.values());
        allValues.addAll(manifest.fixedOutputs().values());
        for (var value : allValues) {
            if (value.resource() != null) {
                queue.add(new Key(value.resource().type(), value.resource().id()));
                queue.addAll(ResourceForkReferences.references(value.resource().type(), value.data()));
            }
        }
        var objects = new LinkedHashMap<Key, Entry>();
        var ids = new LinkedHashSet<String>();
        manifest.media().forEach(file -> ids.add(file.id()));
        var warnings = new LinkedHashSet<>(manifest.warnings());
        while (!queue.isEmpty()) {
            var key = queue.removeFirst();
            if (objects.containsKey(key)) {
                continue;
            }
            var header = records.historical(key.type(), key.id());
            resourceAccess.requireHistorical(header, actor);
            if (!Objects.equals(header.projectId(), document.projectId())) {
                throw new BusinessException(409, "副本来源含其他项目对象，请先修正关联");
            }
            var data = records.raw(key.type(), key.id(), false);
            if (records.find(key.type(), key.id(), false, false).isEmpty()) {
                data.put("deleted", true);
            }
            var resolved = exportMedia.resolve(
                    new Value(key.type().code() + ":" + key.id(), key.type(), null, null, null,
                            new CanvasResourceReference(key.type(), key.id(), (long) header.version()), data, null),
                    actor, ids, warnings);
            objects.put(key, new Entry(key.type(), key.id(), resolved.data()));
            if (objects.size() > 4000) {
                throw new BusinessException(400, "单次独立副本最多复制 4000 个业务对象");
            }
            queue.addAll(ResourceForkReferences.references(key.type(), data));
        }
        var files = ids.stream().sorted().map(id -> {
            var file = media.get(id, actor);
            mediaAccess.ready(file, actor);
            if (file.projectId() != null && !Objects.equals(file.projectId(), document.projectId())) {
                throw new BusinessException(409, "副本来源含其他项目媒体，请先显式导入");
            }
            return file;
        }).toList();
        var entries = objects.values().stream()
                .sorted(Comparator.comparingInt((Entry entry) -> ProjectForkRepository.ORDER.indexOf(entry.type()))
                        .thenComparingLong(Entry::id)).toList();
        var result = new Frozen(manifest, entries, files, primary(manifest, CanvasValueType.SCRIPT),
                primary(manifest, CanvasValueType.STORYBOARD), List.copyOf(warnings), actor);
        if (json.write(result).length() > 16 * 1024 * 1024) {
            throw new BusinessException(400, "副本内容超过 16 MiB，请先按分集拆分项目");
        }
        return result;
    }

    public Preview preview(Frozen source) {
        return new Preview(json.hash(source), source.manifest().documentRevision(),
                source.manifest().graph().nodes().size(), source.objects().size(),
                (int) source.objects().stream().filter(value -> value.data().path("deleted").asBoolean()).count(),
                source.media().size(),
                source.media().stream().mapToLong(file -> file.content().byteSize()).sum(), source.warnings(),
                ownership(source.actor()));
    }

    public Ownership ownership(CanvasActor actor) {
        access.requireActorScope(actor);
        if (actor.teamId() != null) {
            return new Ownership(2, actor.teamId(), teams.selectById(actor.teamId()).getName());
        }
        var user = users.selectById(actor.userId());
        String name = user.getNickname();
        return new Ownership(1, actor.userId(), name == null || name.isEmpty() ? user.getUsername() : name);
    }

    private Long primary(CanvasExportData.Manifest manifest, CanvasValueType type) {
        return manifest.values().stream().filter(value -> value.type() == type && value.resource() != null)
                .map(value -> value.resource().id()).findFirst().orElse(null);
    }
}
