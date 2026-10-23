package com.stonewu.fusion.service.canvas.delivery;

import com.fasterxml.jackson.databind.JsonNode;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.repository.canvas.CanvasGraphRepository;
import com.stonewu.fusion.repository.resource.ResourceReadRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.CanvasMcpConfigMedia;
import com.stonewu.fusion.service.canvas.CanvasResourceCollectionService;
import com.stonewu.fusion.service.canvas.collaboration.CanvasPinnedOutputs;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasDocument;
import com.stonewu.fusion.service.canvas.model.CanvasGraph;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Value;
import com.stonewu.fusion.service.canvas.run.CanvasRunSources;
import com.stonewu.fusion.service.media.MediaAccessPolicy;
import com.stonewu.fusion.service.media.MediaResourceQueries;
import com.stonewu.fusion.service.media.model.MediaResource;
import com.stonewu.fusion.service.resource.AssetResourceQueryService;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.StoryboardResourceQueryService;
import com.stonewu.fusion.service.resource.model.ResourceSnapshot;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.*;

import static com.stonewu.fusion.service.canvas.delivery.CanvasExportData.*;
import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

/**
 * 导出源在一个数据库快照内冻结；只枚举已授权对象，不发起外部媒体下载。
 */
@Component
@RequiredArgsConstructor
public class CanvasExportSources {
    private final CanvasDocumentRepository documents;
    private final CanvasGraphRepository graphs;
    private final CanvasAccessPolicy access;
    private final CanvasRunSources sources;
    private final CanvasResourceCollectionService collections;
    private final ResourceQueryService resources;
    private final AssetResourceQueryService assets;
    private final StoryboardResourceQueryService boards;
    private final MediaResourceQueries media;
    private final MediaAccessPolicy mediaAccess;
    private final CanvasJson json;
    private final CanvasExportMedia exportMedia;
    private final CanvasPinnedOutputs pins;
    private final ResourceReadRepository resourceRecords;
    private final CanvasMcpConfigMedia mcpMedia;

    public Manifest freeze(String canvasId, Options options, CanvasActor actor) {
        var document = documents.require(canvasId, false);
        access.requireView(document, actor);
        if (options.expectedRevision() == null || options.expectedRevision() != document.revision()) {
            throw conflict("画布已变化，请重新预览导出范围");
        }
        var graph = graphs.load(document);
        if (options.nodeIds().size() > 1000 || new LinkedHashSet<>(options.nodeIds()).size() != options.nodeIds()
                .size()) {
            throw invalid("导出节点列表无效");
        }
        Set<String> selected = new LinkedHashSet<>(options.nodeIds());
        if (options.scope() == Scope.SELECTION) {
            if (selected.isEmpty()) {
                throw invalid("请先选择需要导出的节点");
            }
            var nodes = graph.nodes().stream().filter(node -> selected.contains(node.id())).toList();
            if (nodes.size() != selected.size()) {
                throw conflict("部分所选节点已被移除");
            }
            // 分组父节点不在范围时保留世界坐标由模板/分支导入器处理，导出原始布局及父引用。
            graph = new CanvasGraph(graph.schemaVersion(), graph.projectId(), nodes, graph.edges().stream()
                    .filter(edge -> selected.contains(edge.sourceNodeId()) && selected.contains(edge.targetNodeId()))
                    .toList());
        } else if (!selected.isEmpty()) {
            throw invalid("只有所选节点范围允许提供节点列表");
        }
        List<Value> inputs = new ArrayList<>();
        if (options.scope() == Scope.PROJECT) {
            if (document.projectId() == null) {
                throw invalid("个人画布没有项目内容，请选择画布范围");
            }
            inputs.add(sources.business(resources.get(PROJECT, document.projectId(), actor)));
        }
        for (CanvasNode node : graph.nodes()) {
            if (node.resource() != null) {
                inputs.add(sources.business(resources.get(node.resource().type(), node.resource().id(), actor)));
            } else if (!node.kind().storedMedia() || node.config().hasNonNull("resourceId")) {
                inputs.addAll(sources.freeze(document, node, actor, Set.of()).all());
            }
        }
        return freeze(document, graph, document.revision(), inputs, options.format(), actor, true);
    }

    public Manifest freeze(CanvasDocument document, CanvasGraph graph, long revision, List<Value> inputs,
                           Format format, CanvasActor actor) {
        return freeze(document, graph, revision, inputs, format, actor, false);
    }

    private Manifest freeze(CanvasDocument document, CanvasGraph graph, long revision, List<Value> inputs,
                            Format format, CanvasActor actor, boolean expandCurrent) {
        access.requireView(document, actor);
        Map<String, Value> values = new LinkedHashMap<>();
        for (Value value : inputs) {
            if (expandCurrent) {
                add(document, value, actor, values);
            } else {
                authorizeSnapshot(document, value, actor);
                values.putIfAbsent(value.id(), value);
            }
        }
        Set<String> mediaIds = new LinkedHashSet<>();
        Set<String> warnings = new LinkedHashSet<>();
        values.replaceAll((key, value) -> exportMedia.resolve(value, actor, mediaIds, warnings));
        Set<String> sourceMediaIds = new LinkedHashSet<>();
        Set<String> sourceWarnings = new LinkedHashSet<>();
        Map<String, Value> fixedOutputs = new LinkedHashMap<>();
        for (var edge : graph.edges()) {
            if (edge.output() == null) {
                continue;
            }
            Value fixed = pins.resolve(document.id(), edge, actor);
            authorizeSnapshot(document, fixed, actor);
            fixedOutputs.put(edge.id(), exportMedia.resolve(fixed, actor, sourceMediaIds, sourceWarnings));
        }
        collectGraphMedia(graph, sourceMediaIds);
        // 来源图继续保留并检查权限；导出节点只交付接入的内容，画布导出才包含所选图中的素材。
        if (expandCurrent) {
            mediaIds.addAll(sourceMediaIds);
            warnings.addAll(sourceWarnings);
        }
        Set<String> referencedMediaIds = new LinkedHashSet<>(mediaIds);
        referencedMediaIds.addAll(sourceMediaIds);
        Map<String, MediaResource> authorizedMedia = new LinkedHashMap<>();
        for (String id : referencedMediaIds) {
            var file = media.get(id, actor);
            mediaAccess.ready(file, actor);
            if (file.projectId() != null && !Objects.equals(file.projectId(), document.projectId())) {
                throw invalid("画布包含其他项目的媒体，请先显式导入当前项目");
            }
            authorizedMedia.put(id, file);
        }
        if (authorizedMedia.size() > 1000) {
            throw invalid("单次交付最多包含 1000 个媒体文件，请拆分范围");
        }
        List<MediaResource> files = mediaIds.stream().map(authorizedMedia::get).toList();
        long bytes = files.stream().mapToLong(file -> file.content().byteSize()).reduce(0, Math::addExact);
        if ((format == Format.ZIP || format == Format.MEDIA) && bytes > MAX_BYTES) {
            throw invalid("单次媒体交付不能超过 4 GiB，请缩小范围");
        }
        if ((format == Format.ZIP || format == Format.MEDIA) && !warnings.isEmpty()) {
            throw invalid("部分旧媒体尚未登记到素材库，请先完成素材导入后再打包；文稿和 JSON 仍可导出并保留来源说明");
        }
        if (format == Format.MEDIA && files.size() != 1) {
            throw invalid("媒体文件导出需要恰好一个媒体；多个文件请选择交付包");
        }
        if (format == Format.CSV && values.values().stream().noneMatch(v -> v.type() == SHOT)
                && fixedOutputs.values().stream().noneMatch(v -> v.type() == SHOT)) {
            throw invalid("分镜表格范围中没有镜头，请选择分镜或镜头节点，或整个项目");
        }
        var manifest = new Manifest(2, document.title(), revision, graph, List.copyOf(values.values()), files,
                List.copyOf(warnings), fixedOutputs);
        if (json.write(manifest).length() > 8 * 1024 * 1024) {
            throw invalid("交付快照超过 8 MiB，请按分集拆分导出");
        }
        return manifest;
    }

    public void authorize(String canvasId, Manifest manifest, CanvasActor actor) {
        access.requireView(documents.require(canvasId, false), actor);
        for (Value value : manifest.values()) {
            sources.authorize(value, actor, false);
        }
        for (Value value : manifest.fixedOutputs().values()) {
            sources.authorize(value, actor, false);
        }
        for (CanvasNode node : manifest.graph().nodes()) {
            if (node.resource() != null) {
                resources.authorizeHistorical(node.resource().type(), node.resource().id(), actor);
            }
        }
        for (MediaResource file : manifest.media()) {
            var current = media.get(file.id(), actor);
            mediaAccess.ready(current, actor);
            if (file.version() != current.version() || !file.content().contentHash()
                    .equals(current.content().contentHash())) {
                throw conflict("交付引用的媒体版本已变化，请重新创建交付包");
            }
        }
        Set<String> sourceMediaIds = new LinkedHashSet<>();
        collectGraphMedia(manifest.graph(), sourceMediaIds);
        for (Value value : manifest.fixedOutputs().values()) {
            exportMedia.scan(json.object(value), sourceMediaIds, null);
        }
        manifest.media().forEach(file -> sourceMediaIds.remove(file.id()));
        for (String id : sourceMediaIds) {
            mediaAccess.ready(media.get(id, actor), actor);
        }
    }

    private void collectGraphMedia(CanvasGraph graph, Set<String> mediaIds) {
        for (CanvasNode node : graph.nodes()) {
            exportMedia.scan(node.config(), mediaIds, null);
            if (node.kind().storedMedia() && node.config().hasNonNull("resourceId")) {
                mediaIds.add(node.config().path("resourceId").asText());
            }
            mcpMedia.collect(node, mediaIds);
        }
        for (var edge : graph.edges()) {
            if (edge.output() != null && edge.output().resourceId() != null) {
                mediaIds.add(edge.output().resourceId());
            }
        }
    }

    public Preview preview(Manifest manifest, Format format) {
        return new Preview(json.hash(Map.of("format", format, "manifest", manifest)), manifest.documentRevision(),
                manifest.graph().nodes().size(),
                (int) manifest.values().stream().filter(v -> v.resource() != null).count(),
                manifest.media().size(), manifest.media().stream().mapToLong(v -> v.content().byteSize()).sum(),
                manifest.warnings());
    }

    private void authorizeSnapshot(CanvasDocument document, Value value, CanvasActor actor) {
        sources.authorize(value, actor, true);
        if (value.resource() == null) {
            return;
        }
        var ref = value.resource();
        var header = resourceRecords.historical(ref.type(), ref.id());
        if (!Objects.equals(header.projectId(), document.projectId())) {
            throw invalid("业务对象不属于当前画布项目");
        }
        if (ref.version() == null || ref.version() < 0 || ref.version() > header.version()) {
            throw invalid("固定业务结果的版本无效");
        }
    }

    private void add(CanvasDocument document, Value value, CanvasActor actor, Map<String, Value> result) {
        sources.authorize(value, actor, true);
        String key = value.resource() == null ? "value:" + value.id() : value.resource().type()
                .code() + ":" + value.resource().id();
        if (result.containsKey(key)) {
            if (!json.hash(result.get(key).data()).equals(json.hash(value.data()))) {
                throw conflict("导出包含同一对象的不同版本，请明确选择一个版本");
            }
            return;
        }
        result.put(key, value);
        if (result.size() > 4000) {
            throw invalid("单次交付最多包含 4000 个内容对象，请按分集拆分");
        }
        if (value.resource() == null) {
            return;
        }
        var ref = value.resource();
        var current = resources.get(ref.type(), ref.id(), actor);
        if (ref.version() == null || ref.version() != current.header().version()) {
            throw conflict("导出输入的业务版本已变化，请重新读取或运行上游节点");
        }
        if (!Objects.equals(current.header().projectId(), document.projectId())) {
            throw invalid("业务对象不属于当前画布项目");
        }
        if (ref.type() == PROJECT) {
            var script = resources.script(ref.id(), actor);
            if (script != null) {
                add(document, sources.business(resources.get(SCRIPT, script.path("id").asLong(), actor)), actor,
                        result);
            }
            var board = boards.project(ref.id(), actor);
            if (board != null) {
                add(document, sources.business(resources.get(STORYBOARD, board.path("id").asLong(), actor)), actor,
                        result);
            }
            var first = assets.page(ref.id(), null, null, 1, 100, actor);
            long count = first.path("total").asLong();
            if (count > 4000) {
                throw invalid("项目资产超过单次导出上限，请缩小范围");
            }
            for (int page = 1; page <= Math.max(1, (count + 99) / 100); page++) {
                var batch = page == 1 ? first : assets.page(ref.id(), null, null, page, 100, actor);
                for (JsonNode asset : batch.path("records")) {
                    add(document, sources.business(resources.get(ASSET, asset.path("id").asLong(), actor)), actor,
                            result);
                }
            }
        } else {
            var tree = collections.get(document.id(), ref.type(), ref.id(), 3, actor);
            for (var entry : tree.descendants()) {
                Value child = sources.business(new ResourceSnapshot(entry.header(), entry.data()));
                String childKey = child.resource().type().code() + ":" + child.resource().id();
                Value old = result.putIfAbsent(childKey, child);
                if (old != null && !old.resource().version().equals(child.resource().version())) {
                    throw conflict("业务对象版本不一致");
                }
                if (result.size() > 4000) {
                    throw invalid("单次交付最多包含 4000 个内容对象，请按分集拆分");
                }
            }
            // 旧分镜允许镜头暂不属于场次/分集，按实际分镜归属补齐，不能仅依赖三层树而漏掉这些镜头。
            if (ref.type() == STORYBOARD || ref.type() == STORYBOARD_EPISODE) {
                for (var shot : boards.children(SHOT, ref.type(), ref.id(), actor)) {
                    add(document, sources.business(resources.get(SHOT, shot.path("id").asLong(), actor)), actor,
                            result);
                }
            }
        }
    }

}
