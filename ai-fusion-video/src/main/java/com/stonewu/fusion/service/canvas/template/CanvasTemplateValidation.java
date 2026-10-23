package com.stonewu.fusion.service.canvas.template;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.canvas.CanvasGraphValidator;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.CanvasMcpConfigMedia;
import com.stonewu.fusion.service.canvas.capability.CanvasCapabilityRegistry;
import com.stonewu.fusion.service.canvas.collaboration.CanvasProjectForkSources.Entry;
import com.stonewu.fusion.service.canvas.model.CanvasEdgeKind;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Value;
import com.stonewu.fusion.service.media.MediaReference;
import com.stonewu.fusion.service.media.MediaSourceQueries;
import com.stonewu.fusion.service.resource.ResourceForkReferences;
import com.stonewu.fusion.service.resource.ResourceForkReferences.Key;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.nio.charset.StandardCharsets;
import java.util.*;

import static com.stonewu.fusion.service.canvas.template.CanvasTemplateData.*;

/**
 * 文件内容不具有源数据库的任何权限；先验证闭合的本地编号和绑定声明，再允许创建新对象。
 */
@Component
@RequiredArgsConstructor
public class CanvasTemplateValidation {
    private final CanvasGraphValidator graphs;
    private final CanvasCapabilityRegistry registry;
    private final CanvasTemplateResourceValidation fields;
    private final CanvasTemplateProviders providers;
    private final CanvasMcpConfigMedia mcp;
    private final MediaSourceQueries mediaFields;
    private final CanvasJson json;

    public void validate(Template template) {
        if (template == null || !SCHEMA.equals(template.schema()) || template.schemaVersion() != VERSION) {
            throw invalid("不是受支持的画布模板，请使用模板导出文件");
        }
        if (template.title() == null || template.title().isBlank() || template.title()
                .length() > 200 || template.objects() == null
                || template.media() == null || template.providers() == null || template.fixedOutputs() == null) {
            throw invalid("模板基本信息不完整");
        }
        if (json.write(template).getBytes(StandardCharsets.UTF_8).length > MAX_BYTES || template.objects().size() > 4000
                || template.media().size() > 1000 || template.providers().size() > 10000) {
            throw invalid("模板超过大小或对象数量限制");
        }
        graphs.validate(template.graph());
        for (var edge : template.graph().edges()) {
            if (edge.output() != null) {
                throw invalid("模板不能携带原数据库的运行或固定快照编号");
            }
        }
        Map<Key, Entry> objects = new HashMap<>();
        Set<String> usedMedia = new HashSet<>();
        for (var entry : template.objects()) {
            if (entry == null || entry.type() == null || entry.id() > 4000 || objects.putIfAbsent(
                    new Key(entry.type(), entry.id()), entry) != null) {
                throw invalid("模板业务编号重复或无效");
            }
            fields.validate(entry.type(), entry.id(), entry.data());
            businessMedia(entry.type(), entry.data(), usedMedia);
        }
        for (var entry : template.objects()) {
            for (var reference : ResourceForkReferences.references(entry.type(), entry.data())) {
                if (!objects.containsKey(reference)) {
                    throw invalid("模板缺少业务依赖：" + reference.type().code() + "/" + reference.id());
                }
            }
        }
        var roots = template.objects().stream().filter(entry -> entry.type() == CanvasValueType.PROJECT).toList();
        if (roots.size() > 1 || (template.graph().projectId() == null ? !roots.isEmpty()
                : roots.size() != 1 || roots.getFirst().id() != template.graph().projectId() || roots.getFirst().data()
                .path("deleted").asBoolean())) {
            throw invalid("模板项目归属不一致");
        }
        primary(objects, CanvasValueType.SCRIPT, template.primaryScript());
        primary(objects, CanvasValueType.STORYBOARD, template.primaryStoryboard());
        Map<String, CanvasNode> nodes = new HashMap<>();
        for (var node : template.graph().nodes()) {
            nodes.put(node.id(), node);
            if (node.kind() == CanvasNodeKind.SELECTION && node.config().hasNonNull("selectedOutputId")) {
                throw invalid("模板候选节点应在新运行中重新选择结果");
            }
            if (node.resource() != null) {
                var entry = objects.get(new Key(node.resource().type(), node.resource().id()));
                if (entry == null || entry.data().path("deleted").asBoolean() || node.resource()
                        .version() != entry.data().path("version").longValue()) {
                    throw invalid("模板节点没有对应的当前业务正文");
                }
            }
            scan(node.config(), usedMedia);
            mcp.collect(node, usedMedia);
            if (node.kind().storedMedia() && node.config().hasNonNull("resourceId")) {
                usedMedia.add(node.config().path("resourceId").asText());
            }
        }
        Map<String, Media> files = new HashMap<>();
        for (var file : template.media()) {
            if (file == null || !uuid(file.id()) || files.putIfAbsent(file.id(),
                    file) != null || file.name() == null || file.name().length() > 1024
                    || file.content() == null || file.content().kind() == null || file.content()
                    .byteSize() < 0 || file.content().contentHash() == null
                    || !file.content().contentHash().matches("[0-9a-f]{64}")) {
                throw invalid("模板素材声明无效");
            }
        }
        for (var node : template.graph().nodes()) {
            if (node.kind().storedMedia() && node.config().hasNonNull("resourceId")) {
                var file = files.get(node.config().path("resourceId").asText());
                var type = registry.require(node.kind()).outputs().getFirst().resolveType(node.config());
                if (file == null || !type.code().equals(file.content().kind().name().toLowerCase(Locale.ROOT))) {
                    throw invalid("模板媒体节点与素材类型不一致");
                }
            }
        }
        Map<String, String> values = new HashMap<>();
        for (var fixed : template.fixedOutputs().entrySet()) {
            var edge = template.graph().edges().stream().filter(candidate -> candidate.id().equals(fixed.getKey()))
                    .findFirst().orElseThrow(() -> invalid("固定结果没有对应连线"));
            if (edge.kind() != CanvasEdgeKind.EXECUTION) {
                throw invalid("只有执行连线可以固定结果");
            }
            var node = nodes.get(edge.sourceNodeId());
            var port = registry.require(node.kind()).outputs().stream()
                    .filter(candidate -> candidate.id().equals(edge.sourcePort())).findFirst().orElseThrow();
            if (fixed.getValue() == null || port.resolveType(node.config()) != fixed.getValue().type()) {
                throw invalid("模板固定结果与来源端口类型不一致");
            }
            value(fixed.getValue(), objects, files, usedMedia, values, 0);
        }
        if (!files.keySet().equals(usedMedia)) {
            throw invalid("模板素材声明与实际引用不一致");
        }
        providerUses(template, nodes);
    }

    private void primary(Map<Key, Entry> objects, CanvasValueType type, Long id) {
        if (id == null) {
            if (objects.values().stream()
                    .anyMatch(entry -> entry.type() == type && !entry.data().path("deleted").asBoolean())) {
                throw invalid("模板必须声明当前的主剧本与主分镜");
            }
            return;
        }
        var entry = objects.get(new Key(type, id));
        if (entry == null || entry.data().path("deleted").asBoolean()) {
            throw invalid("模板主剧本或主分镜不存在");
        }
    }

    private void businessMedia(CanvasValueType type, ObjectNode data, Set<String> used) {
        scan(data, used);
        for (String field : mediaFields.fields(type)) {
            var value = data.at("/" + field.replace('.', '/'));
            if (value.isMissingNode() || value.isNull() || value.isTextual() && value.asText().isBlank()) {
                continue;
            }
            if (!value.isTextual() || !MediaReference.managed(value.asText())) {
                throw invalid("模板的业务素材必须使用明确的素材绑定");
            }
            used.add(MediaReference.id(value.asText()));
        }
    }

    private void scan(JsonNode node, Set<String> used) {
        if (node == null || node.isNull()) {
            return;
        }
        if (node.isTextual() && MediaReference.managed(node.asText())) {
            used.add(MediaReference.id(node.asText()));
        } else if (node.isContainerNode()) {
            node.forEach(value -> scan(value, used));
        }
    }

    private void value(Value value, Map<Key, Entry> objects, Map<String, Media> files, Set<String> used,
                       Map<String, String> hashes, int depth) {
        if (depth > 4 || !uuid(value.id()) || value.type() == null || value.sourceNodeRunId() != null) {
            throw invalid("模板固定结果身份无效");
        }
        String previous = hashes.putIfAbsent(value.id(), json.hash(value));
        if (previous != null && !previous.equals(json.hash(value))) {
            throw invalid("同一模板结果编号不能代表不同内容");
        }
        if (value.resource() != null) {
            var ref = value.resource();
            if (ref.type() == null || ref.id() == null || ref.id() <= 0) {
                throw invalid("模板固定业务编号无效");
            }
            var entry = objects.get(new Key(ref.type(), ref.id()));
            if (entry == null || value.type() != ref.type() || ref.version() == null || ref.version() < 0 || ref.version() > entry.data()
                    .path("version").longValue()
                    || value.data() == null || value.data().path("version")
                    .asLong(-1) != ref.version() || value.mediaId() != null) {
                throw invalid("模板固定业务版本不完整");
            }
            var data = value.data().deepCopy();
            data.remove("_canvasVideos");
            fields.validate(ref.type(), ref.id(), data);
            businessMedia(ref.type(), data, used);
            if (value.data().has("_canvasVideos")) {
                if (!value.data().path("_canvasVideos").isArray()) {
                    throw invalid("模板固定片段必须是数组");
                }
                for (var nested : value.data().path("_canvasVideos")) {
                    var video = json.read(nested.toString(), Value.class);
                    if (video.type() != CanvasValueType.VIDEO || video.mediaId() == null) {
                        throw invalid("模板固定片段必须绑定视频素材");
                    }
                    value(video, objects, files, used, hashes, depth + 1);
                }
            }
        } else if (value.type().isBusinessResource()) {
            throw invalid("固定业务内容缺少引用");
        }
        if (value.mediaId() != null) {
            var file = files.get(value.mediaId());
            if (file == null || !value.type().code().equals(file.content().kind().name().toLowerCase(Locale.ROOT))
                    || value.mediaVersion() == null || value.mediaVersion() != 0 || value.data() != null) {
                throw invalid("模板固定素材类型或版本无效");
            }
            used.add(value.mediaId());
        } else if (value.mediaVersion() != null) {
            throw invalid("模板素材版本缺少素材声明");
        }
        if (value.type() == CanvasValueType.TEXT && value.text() == null) {
            throw invalid("模板固定文本不能为空值");
        }
        if (value.resource() == null && value.mediaId() == null && (value.type() != CanvasValueType.TEXT || value.data() != null)) {
            throw invalid("模板固定媒体缺少绑定，或文本含未声明的元数据");
        }
        if (value.text() != null && value.text().length() > 8 * 1024 * 1024) {
            throw invalid("模板固定文本过长");
        }
    }

    private void providerUses(Template template, Map<String, CanvasNode> nodes) {
        Set<String> ids = new HashSet<>();
        Set<Use> uses = new HashSet<>();
        for (var slot : template.providers()) {
            if (slot == null || !uuid(slot.id()) || !ids.add(
                    slot.id()) || slot.kind() == null || slot.label() == null || slot.label().length() > 256
                    || slot.uses() == null || slot.uses().isEmpty() || slot.uses().size() > 10000) {
                throw invalid("模板提供商声明无效");
            }
            for (var use : slot.uses()) {
                var node = use == null ? null : nodes.get(use.nodeId());
                if (node == null || use.field() == null || !uses.add(use) || providers.kind(node, use) != slot.kind()) {
                    throw invalid("模板提供商字段重复或类型不一致");
                }
            }
        }
        for (var node : template.graph().nodes()) {
            if (!providers.references(node).isEmpty()) {
                throw invalid("模板不得保留原环境的提供商编号");
            }
        }
    }

    private static boolean uuid(String id) {
        return id != null && id.matches("[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}");
    }

    private static BusinessException invalid(String text) {
        return new BusinessException(400, text);
    }
}
