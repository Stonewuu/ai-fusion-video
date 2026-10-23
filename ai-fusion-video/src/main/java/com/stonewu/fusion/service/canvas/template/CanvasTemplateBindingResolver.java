package com.stonewu.fusion.service.canvas.template;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.CanvasModelGraphPolicy;
import com.stonewu.fusion.service.canvas.capability.CanvasModelCatalog;
import com.stonewu.fusion.service.canvas.collaboration.CanvasProjectForkSources;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasGraph;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.run.mcp.CanvasMcpContracts;
import com.stonewu.fusion.service.media.MediaAccessPolicy;
import com.stonewu.fusion.service.media.MediaResourceQueries;
import com.stonewu.fusion.service.media.model.MediaResource;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.*;
import java.util.stream.Collectors;

import static com.stonewu.fusion.service.canvas.template.CanvasTemplateData.*;

@Component
@RequiredArgsConstructor
public class CanvasTemplateBindingResolver {
    private final CanvasTemplateProviders providers;
    private final CanvasModelCatalog models;
    private final CanvasModelGraphPolicy modelPolicy;
    private final CanvasMcpContracts contracts;
    private final MediaResourceQueries media;
    private final MediaAccessPolicy mediaAccess;
    private final CanvasProjectForkSources scopes;
    private final CanvasJson json;

    public record Resolved(Template template, CanvasGraph graph, Map<String, MediaResource> media,
                           List<String> notices, Map<String, String> providerNames, CanvasActor actor) {
    }

    public record Preview(String hash, int nodes, int businessObjects, int mediaFiles, List<String> notices,
                          Map<String, String> providerNames, CanvasProjectForkSources.Ownership ownership) {
    }

    public Resolved resolve(Template template, Bindings choices, CanvasActor actor) {
        if (choices == null || choices.media() == null || choices.providers() == null) {
            throw invalid("请提供模板的完整绑定选择");
        }
        scopes.ownership(actor);
        Set<String> expectedMedia = template.media().stream().map(Media::id).collect(Collectors.toSet());
        Set<String> expectedProviders = template.providers().stream().map(Provider::id).collect(Collectors.toSet());
        if (!expectedMedia.equals(choices.media().keySet()) || !expectedProviders.equals(
                choices.providers().keySet())) {
            throw invalid("请选择所有模板素材和提供商，不能附加未声明的绑定");
        }
        Map<String, MediaResource> files = new LinkedHashMap<>();
        List<String> notices = new ArrayList<>();
        for (var slot : template.media()) {
            var choice = choice(choices.media(), slot.id());
            if (choice.version() == null || choice.version() < 0) {
                throw invalid("素材绑定缺少明确版本");
            }
            var file = media.get(choice.id(), actor);
            mediaAccess.ready(file, actor);
            if ((long) file.version() != choice.version()) {
                throw invalid("所选素材版本已变化，请重新选择");
            }
            if (file.content().kind() != slot.content().kind()) {
                throw invalid("所选素材类型不匹配：" + slot.name());
            }
            if (!file.content().contentHash().equals(slot.content().contentHash())) {
                notices.add("“" + slot.name() + "”将使用所选的替换素材“" + file.name() + "”。");
            }
            files.put(slot.id(), file);
        }
        var nodes = template.graph().nodes().stream()
                .map(node -> new CanvasNode(node.id(), node.kind(), node.schemaVersion(), node.title(), node.layout(),
                        node.config().deepCopy(), node.resource(), 0, 0)).toList();
        Map<String, CanvasNode> indexed = new HashMap<>();
        nodes.forEach(node -> indexed.put(node.id(), node));
        Map<String, String> names = new LinkedHashMap<>();
        for (var slot : template.providers()) {
            var choice = choice(choices.providers(), slot.id());
            String name;
            if (slot.kind() == ProviderKind.MCP) {
                if (choice.version() == null || choice.version() < 0) {
                    throw invalid("请选择 MCP 操作的明确版本");
                }
                var contract = contracts.version(choice.id(), choice.version(), actor.userId());
                contracts.verify(contract, actor.userId());
                name = contract.name();
            } else {
                long id = number(choice.id());
                if (choice.version() != null) {
                    throw invalid("此提供商绑定不接受额外版本字段");
                }
                if (slot.kind() == ProviderKind.AUDIO_API) {
                    models.requireAudio(id);
                    name = models.audioServices().stream().filter(item -> item.id() == id).findFirst().orElseThrow()
                            .name();
                } else if (slot.kind() == ProviderKind.WORKFLOW) {
                    name = "工作流版本 #" + id;
                } else {
                    var model = models.require(id);
                    String kind = switch (slot.kind()) {
                        case TEXT_MODEL -> "text";
                        case IMAGE_MODEL -> "image";
                        case VIDEO_MODEL -> "video";
                        case COMFYUI_MODEL -> "comfyui";
                        default -> throw invalid("提供商类型无效");
                    };
                    if (!Objects.equals(model.kind(), kind)) {
                        throw invalid("所选模型类型与模板不匹配：" + slot.label());
                    }
                    name = model.name();
                }
            }
            names.put(slot.id(), name);
            for (var use : slot.uses()) {
                var config = providers.config(indexed.get(use.nodeId()), use.batch());
                if (slot.kind() == ProviderKind.MCP) {
                    config.put(use.field(), choice.id());
                    config.put("contractVersion", choice.version());
                } else {
                    config.put(use.field(), number(choice.id()));
                }
            }
        }
        var graph = new CanvasGraph(1, template.graph().projectId(), nodes, template.graph().edges());
        modelPolicy.validateDraft(graph);
        for (var node : nodes) {
            if (node.kind() == CanvasNodeKind.MCP_TOOL && node.config().hasNonNull("contractId")) {
                contracts.freeze(node, actor);
            }
            if (node.kind() == CanvasNodeKind.FOREACH && node.config().path("operationKind").asText().equals("mcpTool")
                    && node.config().path("operationConfig").hasNonNull("contractId")) {
                var body = new CanvasNode(node.id(), CanvasNodeKind.MCP_TOOL, 1, node.title(), node.layout(),
                        providers.config(node, true), null, 0, 0);
                contracts.freeze(body, actor);
            }
        }
        return new Resolved(template, graph, Map.copyOf(files), List.copyOf(notices), Map.copyOf(names), actor);
    }

    public Preview preview(Resolved resolved) {
        return new Preview(json.hash(resolved), resolved.graph().nodes().size(), resolved.template().objects().size(),
                resolved.media().size(),
                resolved.notices(), resolved.providerNames(), scopes.ownership(resolved.actor()));
    }

    private Binding choice(Map<String, Binding> choices, String id) {
        var result = choices.get(id);
        if (result == null || result.id() == null || result.id().isBlank() || result.id().length() > 128) {
            throw invalid("模板绑定编号无效");
        }
        return result;
    }

    private long number(String value) {
        if (!value.matches("[1-9][0-9]{0,15}")) {
            throw invalid("模型或服务编号无效");
        }
        long id = Long.parseLong(value);
        if (id > 9_007_199_254_740_991L) {
            throw invalid("模型或服务编号超出范围");
        }
        return id;
    }

    private BusinessException invalid(String text) {
        return new BusinessException(400, text);
    }
}
