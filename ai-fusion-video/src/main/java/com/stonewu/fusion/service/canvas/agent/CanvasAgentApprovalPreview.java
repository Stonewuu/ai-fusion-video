package com.stonewu.fusion.service.canvas.agent;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.repository.ai.AgentRunRepository;
import com.stonewu.fusion.service.ai.agentscope.context.AgentScopeRuntimeContextRequest;
import com.stonewu.fusion.service.ai.run.AgentConfirmationPreviewProvider;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.CanvasQueryService;
import com.stonewu.fusion.service.canvas.capability.CanvasCapabilityRegistry;
import com.stonewu.fusion.service.canvas.capability.CanvasWritebackFields;
import com.stonewu.fusion.service.canvas.model.*;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.State;
import com.stonewu.fusion.service.canvas.run.CanvasRunEnvironment;
import com.stonewu.fusion.service.canvas.run.CanvasRunRepository;
import com.stonewu.fusion.service.canvas.run.CanvasRunService;
import com.stonewu.fusion.service.canvas.run.CanvasRunSources;
import com.stonewu.fusion.service.canvas.run.batch.CanvasBatchItems;
import com.stonewu.fusion.service.canvas.run.mcp.CanvasMcpPlanner;
import com.stonewu.fusion.service.resource.ResourceAccessPolicy;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.stereotype.Component;

import java.time.Instant;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

@Component
@RequiredArgsConstructor
public class CanvasAgentApprovalPreview implements AgentConfirmationPreviewProvider {
    private static final Set<String> SUPPORTED = Set.of("run_canvas_nodes", "apply_canvas_operations",
            "edit_canvas_resource",
            "bind_canvas_output", "confirm_canvas_writeback", "undo_canvas_command", "review_canvas_batch",
            "arrange_canvas_nodes");
    private final AgentRunRepository agents;
    private final CanvasQueryService canvases;
    private final CanvasRunRepository runs;
    private final ObjectProvider<CanvasRunService> runService;
    private final CanvasJson json;
    private final CanvasRunEnvironment environments;
    private final CanvasCapabilityRegistry capabilities;
    private final ResourceQueryService resources;
    private final ResourceAccessPolicy resourceAccess;
    private final CanvasRunSources sources;
    private final CanvasMcpPlanner mcp;
    private final CanvasApprovalText display;

    @Override
    public boolean supports(String name) {
        return SUPPORTED.contains(name);
    }

    @Override
    public JsonNode preview(String name, JsonNode input, AgentScopeRuntimeContextRequest runtime) {
        var scope = runtime.toolExecution() == null ? null : runtime.toolExecution().canvasScope();
        if (scope == null) {
            throw new BusinessException(400, "画布审批缺少本轮作用域");
        }
        var agent = agents.findRun(runtime.run().runId());
        if (agent == null || !Objects.equals(agent.getUserId(), runtime.authenticatedUser().userId())) {
            throw new BusinessException(403, "无法读取此任务的审批范围");
        }
        var actor = new CanvasActor(agent.getUserId(), agent.getActorTeamId());
        var snapshot = canvases.snapshot(scope.canvasId(), actor);
        var nodes = snapshot.graph().nodes().stream().collect(Collectors.toMap(CanvasNode::id, Function.identity()));
        var result = json.object(Map.of("title", title(name), "summary", snapshot.document().title()));
        var rows = result.putArray("rows");
        var changes = result.putArray("changes");
        var warnings = result.putArray("warnings");
        if (name.equals("apply_canvas_operations")) {
            CanvasAgentSchema.validate(input,
                    CanvasAgentSchema.schema(Map.of("operations", CanvasAgentGraphSchema.operations()),
                            "operations"), "画布修改");
        }
        if (name.equals("run_canvas_nodes")) {
            var stored = runs.plan(required(input, "planId"));
            var plan = stored.value();
            if (!plan.canvasId().equals(scope.canvasId()) || !plan.actor().equals(actor)) {
                throw new BusinessException(403, "运行计划不属于当前画布和身份");
            }
            if (!stored.hash().equals(required(input, "planHash")) || !plan.expiresAt().isAfter(Instant.now())) {
                throw new BusinessException(409, "运行计划已过期或内容变化，请重新预览");
            }
            row(rows, "运行范围", plan.cost().nodes() + " 个节点");
            if (plan.cost().images() > 0) {
                row(rows, "图片上限", plan.cost().images() + " 张");
            }
            if (plan.cost().videoSeconds() > 0) {
                row(rows, "视频上限", plan.cost().videoSeconds() + " 秒");
            }
            if (plan.cost().batchItems() > 0) {
                row(rows, "批量条目", String.valueOf(plan.cost().batchItems()));
            }
            if (plan.cost().agentRuns() > 0) {
                row(rows, "智能体任务上限", plan.cost().agentRuns() + " 个（含主任务和子任务）");
            }
            if (plan.cost().mcpCalls() > 0) {
                row(rows, "外部 MCP 调用上限", plan.cost().mcpCalls() + " 次");
            }
            if (plan.cost().amount() != null) {
                row(rows, "预计费用", plan.cost().amount());
            }
            row(rows, "同时处理上限", plan.budget().maxParallel() + " 个节点");
            for (var node : plan.nodes()) {
                change(changes, node.action().equals("REUSE") ? "复用" : "运行", node.node().title(),
                        capabilities.require(node.node().kind()).label() + (node.action()
                                .equals("REUSE") ? " · 使用已有结果" : configurationDetail(node.node().config(),
                                actor)));
                var external = mcp.describe(node);
                if (external != null) {
                    row(rows, external.name(), external.serverName() + " · "
                            + (external.effect().equals("READ_ONLY") ? "只读" : external.effect()
                            .equals("DESTRUCTIVE") ? "可能删除或覆盖外部内容" : "修改外部内容")
                            + " · 最多 " + external.maximumCalls() + " 次 · 超时 " + external.timeoutSeconds() + " 秒");
                }
            }
            warnings.add("将按以上范围运行；内容、模型或运行上限改变后，需要重新确认。");
        } else if (name.equals("review_canvas_batch")) {
            var run = runs.run(required(input, "canvasRunId"), false);
            if (!run.canvasId().equals(scope.canvasId()) || !run.actor().equals(actor)) {
                throw new BusinessException(403, "批量运行不属于当前画布和身份");
            }
            var node = runs.node(required(input, "nodeRunId"), false);
            if (!node.runId().equals(run.id()) || !"BATCH".equals(
                    node.externalKind()) || node.status() != State.WAITING_CONFIRMATION
                    || node.revision() != input.path("expectedRevision").asLong(-1)) {
                throw new BusinessException(409, "批量等待状态已变化，请重新读取");
            }
            var items = CanvasBatchItems.latest(runs.nodes(run.id()), node.id());
            var failed = items.stream().filter(item -> item.status() != State.SUCCEEDED).toList();
            row(rows, "批量节点", node.plan().node().title());
            row(rows, "处理方式", switch (required(input, "action")) {
                case "RETRY_FAILED" -> "重试失败项";
                case "USE_SUCCEEDED" -> "采用已成功的结果";
                case "FINISH" -> "结束批次";
                default -> throw new BusinessException(400, "请选择有效的批量处理方式");
            });
            row(rows, "处理说明", required(input, "note"));
            if (input.path("action").asText().equals("RETRY_FAILED")) {
                row(rows, "新增任务上限", String.valueOf(
                        failed.stream().mapToInt(item -> 1 + environments.extraAgentRuns(item.plan().node())).sum()));
                row(rows, "新增生成上限", failed.stream().mapToInt(item -> item.plan().imageCount()).sum() + " 张图片 · "
                        + failed.stream().mapToInt(item -> item.plan().videoSeconds()).sum() + " 秒视频");
                for (var item : failed) {
                    change(changes, "重试", "第 " + (item.itemIndex() + 1) + " 项",
                            "第 " + (item.attempt() + 1) + " 次处理");
                }
                warnings.add("会发起新的执行，仍受原运行预算约束；不会重试已成功的项目或尚未确认结束的请求。");
            } else {
                row(rows, "保留结果",
                        items.stream().filter(item -> !item.outputs().all().isEmpty()).count() + " 项已有结果");
            }
        } else if (name.equals("arrange_canvas_nodes")) {
            String mode = switch (required(input, "mode")) {
                case "arrange" -> "按连线自动整理";
                case "left" -> "左对齐";
                case "top" -> "顶部对齐";
                case "horizontal" -> "横向排列";
                case "vertical" -> "纵向排列";
                default -> throw new BusinessException(400, "排列方式无效");
            };
            result.put("title", "确认排列节点");
            row(rows, "排列方式", mode);
            for (var id : input.path("nodeIds")) {
                var node = nodes.get(id.asText());
                if (node == null) {
                    throw new BusinessException(404, "待排列节点不存在");
                }
                change(changes, node.layout().locked() ? "保持原位" : "排列", node.title(),
                        capabilities.require(node.kind()).label());
            }
            warnings.add("仅调整节点位置；锁定节点保持原位，正文和连线不变。");
        } else if (name.equals("apply_canvas_operations")) {
            Map<String, String> titles = new HashMap<>();
            nodes.forEach((id, node) -> titles.put(id, node.title()));
            for (var operation : input.path("operations")) {
                if (operation.path("type").asText().equals("canvas.node.create")) {
                    titles.put(operation.path("node").path("id").asText(),
                            operation.path("node").path("title").asText());
                }
            }
            for (var operation : input.path("operations")) {
                String type = operation.path("type").asText();
                String id = operation.path("nodeId").asText();
                String label = operation.path("node").path("title").asText(titles.getOrDefault(id, id));
                change(changes, action(type), label.isBlank() ? "连线" : label,
                        operationDetail(operation, titles, actor));
            }
        } else if (name.equals("bind_canvas_output")) {
            previewBinding(input, snapshot, actor, rows, changes, warnings);
        } else if (name.equals("confirm_canvas_writeback")) {
            var storedRun = runs.run(required(input, "canvasRunId"), false);
            if (!storedRun.canvasId().equals(scope.canvasId())) {
                throw new BusinessException(403, "提案不属于当前画布");
            }
            var view = runService.getObject().view(storedRun.id(), actor);
            var node = view.nodes().stream().filter(item -> item.id().equals(required(input, "nodeRunId"))).findFirst()
                    .orElseThrow(() -> new BusinessException(404, "写回节点不存在"));
            var proposal = node.writeback();
            if (proposal == null || !proposal.path("hash").asText().equals(required(input, "proposalHash"))
                    || node.revision() != input.path("expectedRevision").asLong(-1)) {
                throw new BusinessException(409, "写回提案或版本已变化，请重新读取");
            }
            change(changes, "写入", proposal.path("targetTitle").asText(node.title()),
                    proposal.path("field").path("label").asText());
            var valueType = CanvasValueType.fromCode(proposal.path("field").path("valueType").asText());
            row(rows, "修改前", display.describeValue(proposal.path("before"), valueType, actor));
            row(rows, "修改后", display.describeValue(proposal.path("value"), valueType, actor));
            if (proposal.hasNonNull("notice")) {
                warnings.add(proposal.path("notice").asText());
            }
            for (var removed : proposal.path("removedResources")) {
                change(changes, "删除", removed.path("title").asText(), "关联内容也会移除");
            }
        } else if (name.equals("undo_canvas_command")) {
            row(rows, "操作范围", "撤销选定的画布修改");
            warnings.add("如果其他人继续修改过这些内容，将先提示冲突。已经生成的媒体和已保存的项目内容会保留。");
        } else {
            String id = input.path(name.equals("bind_canvas_output") ? "targetNodeId" : "nodeId").asText();
            var node = nodes.get(id);
            if (!"create".equals(input.path("action").asText()) && node == null) {
                throw new BusinessException(404, "目标节点不存在");
            }
            if (node != null && node.resource() == null) {
                throw new BusinessException(400, "请选择已关联项目内容的节点");
            }
            var resourceType = node == null ? CanvasValueType.fromCode(
                    input.path("resourceType").asText()) : node.resource().type();
            change(changes, name.equals("bind_canvas_output") ? "采用结果" : action(input.path("action").asText()),
                    node == null ? input.path("title").asText() : node.title(),
                    "delete".equals(input.path("action").asText())
                            ? "删除项目中的这项内容" : display.describeResourceFields(input.path("fields"),
                            resourceType, actor));
            if (name.equals("edit_canvas_resource") && input.path("action").asText().equals("patch")
                    && node != null && node.resource() != null) {
                var ref = node.resource();
                var current = resources.get(ref.type(), ref.id(), actor);
                resourceAccess.requireEdit(current.header(), actor);
                if (current.header().version() != input.path("expectedVersion").asLong(-1)) {
                    throw new BusinessException(409, "业务内容已变化，请重新读取后确认");
                }
                for (var field : CanvasWritebackFields.list()) {
                    if (field.targetType() == ref.type() && field.valueType() == CanvasValueType.TEXT
                            && input.path("fields").has(field.path())) {
                        row(rows, field.label() + " · 修改前",
                                display.describe(current.data().path(field.path()), actor));
                        row(rows, field.label() + " · 修改后",
                                display.describe(input.path("fields").path(field.path()), actor));
                    }
                }
            }
            if ("delete".equals(input.path("action").asText())) {
                warnings.add("这会从项目中删除该内容及其关联内容，其他页面也会同步更新。");
            }
            if ("replaceSource".equals(input.path("action").asText()) || "script:rawContent".equals(
                    input.path("binding").asText())) {
                warnings.add("替换剧本原文将重置解析结果，移除原分集和场次。");
            }
        }
        return result;
    }

    private void previewBinding(JsonNode input, CanvasSnapshot snapshot, CanvasActor actor,
                                ArrayNode rows, ArrayNode changes, ArrayNode warnings) {
        var run = runs.run(required(input, "canvasRunId"), false);
        if (!run.canvasId().equals(snapshot.document().id())) {
            throw new BusinessException(403, "结果不属于当前画布");
        }
        var target = snapshot.graph().nodes().stream().filter(node -> node.id().equals(required(input, "targetNodeId")))
                .findFirst().orElseThrow(() -> new BusinessException(404, "写入节点不存在"));
        if (target.resource() == null) {
            throw new BusinessException(400, "请选择明确的业务写入目标");
        }
        var value = runService.getObject().view(run.id(), actor).nodes().stream()
                .flatMap(node -> node.outputs().all().stream())
                .filter(item -> item.id().equals(required(input, "outputId"))).findFirst()
                .orElseThrow(() -> new BusinessException(404, "运行输出尚未发布或不可访问"));
        sources.authorize(value, actor, true);
        var ref = target.resource();
        var current = resources.get(ref.type(), ref.id(), actor);
        resourceAccess.requireEdit(current.header(), actor);
        if (!Objects.equals(current.header().projectId(), snapshot.document().projectId())
                || current.header().version() != input.path("expectedVersion").asInt(-1)) {
            throw new BusinessException(409, "写入范围或目标版本已变化，请重新读取并确认");
        }
        var field = CanvasWritebackFields.require(json.object(Map.of("binding", required(input, "binding"),
                "targetType", ref.type().code(), "valueType", value.type().code())));
        JsonNode before = current.data();
        for (String segment : field.path().split("\\.")) {
            before = before.path(segment);
        }
        change(changes, "写入", target.title(), field.label());
        row(rows, "写入前", display.describeValue(before, field.valueType(), actor));
        row(rows, "写入后", value.text() != null ? value.text() : display.describe(
                json.object(Map.of("resourceId", value.mediaId())), actor));
        if (field.key().equals("script:rawContent")) {
            warnings.add("替换剧本原文会重置解析状态、摘要和角色信息，并移除以下现有分集与场次。");
            for (var episode : resources.scriptChildren(CanvasValueType.SCRIPT_EPISODE, ref.id(), actor)) {
                change(changes, "删除", episode.path("title").asText(), "剧本分集");
                for (var scene : resources.scriptChildren(CanvasValueType.SCRIPT_SCENE, episode.path("id").asLong(),
                        actor)) {
                    change(changes, "删除", scene.path("sceneHeading").asText(), "剧本场次");
                }
            }
        }
    }

    private String operationDetail(JsonNode operation, Map<String, String> titles, CanvasActor actor) {
        var node = operation.path("node");
        var edge = operation.path("edge");
        return switch (operation.path("type").asText()) {
            case "canvas.node.create" ->
                    capabilities.require(CanvasNodeKind.fromCode(node.path("kind").asText())).label()
                            + configurationDetail(node.path("config"), actor);
            case "canvas.node.patch" ->
                    (operation.hasNonNull("title") ? "名称改为「" + operation.path("title").asText() + "」" : "更新节点")
                            + configurationDetail(operation.path("configPatch"), actor);
            case "canvas.node.delete" -> "移除画布节点，保留它引用的项目内容";
            case "canvas.node.layout" -> "调整位置、尺寸或分组；不修改节点内容";
            case "canvas.node.bind" -> "关联内容：" + display.resourceName(
                    CanvasValueType.fromCode(operation.path("resource").path("type").asText()),
                    operation.path("resource").path("id").asLong(), actor);
            case "canvas.node.assert" -> "确认节点内容未被其他人修改";
            case "canvas.edge.create", "canvas.edge.replace" -> edgeKindLabel(edge.path("kind").asText()) + " · "
                    + titles.getOrDefault(edge.path("sourceNodeId").asText(), edge.path("sourceNodeId").asText())
                    + " → " + titles.getOrDefault(edge.path("targetNodeId").asText(),
                    edge.path("targetNodeId").asText());
            case "canvas.edge.delete" -> "移除节点间的连接，保留两个节点";
            default -> throw new BusinessException(400, "不支持的画布修改类型");
        };
    }

    private static String edgeKindLabel(String code) {
        return switch (CanvasEdgeKind.fromCode(code)) {
            case EXECUTION -> "执行连线";
            case REFERENCE -> "来源引用";
            case ANNOTATION -> "注释连线";
        };
    }

    private String configurationDetail(JsonNode config, CanvasActor actor) {
        if (!config.isObject() || config.isEmpty()) {
            return "";
        }
        return "\n" + display.describe(config, actor);
    }

    private void row(ArrayNode rows, String label, String value) {
        rows.add(json.object(Map.of("label", label, "value", value == null ? "未配置" : value)));
    }

    private void change(ArrayNode changes, String action, String object, String detail) {
        changes.add(json.object(Map.of("action", action, "object", object, "detail", detail)));
    }

    private static String required(JsonNode value, String field) {
        if (!value.path(field).isTextual() || value.path(field).asText().isBlank()) {
            throw new BusinessException(400, "缺少参数：" + field);
        }
        return value.path(field).asText();
    }

    private static String title(String name) {
        return switch (name) {
            case "run_canvas_nodes" -> "确认运行这些节点";
            case "edit_canvas_resource" -> "确认项目内容修改";
            case "bind_canvas_output", "confirm_canvas_writeback" -> "确认采用创作结果";
            case "undo_canvas_command" -> "确认撤销画布操作";
            case "review_canvas_batch" -> "确认处理批量结果";
            default -> "确认画布修改";
        };
    }

    private static String action(String type) {
        if (type.endsWith("delete")) {
            return "删除";
        }
        if (type.endsWith("create")) {
            return "新增";
        }
        if (type.endsWith("layout")) {
            return "移动";
        }
        if (type.endsWith("bind")) {
            return "绑定";
        }
        return "修改";
    }
}
