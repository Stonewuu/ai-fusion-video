package com.stonewu.fusion.service.canvas.agent;

import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.ai.tool.generation.GetGenerationTaskToolExecutor;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.CanvasQueryService;
import com.stonewu.fusion.service.canvas.CanvasUndoService;
import com.stonewu.fusion.service.canvas.CanvasWorkspaceService;
import com.stonewu.fusion.service.canvas.command.CanvasUndoCommand;
import com.stonewu.fusion.service.canvas.command.CanvasWorkspaceCommands.CheckpointCreate;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Budget;
import com.stonewu.fusion.service.generation.job.GenerationJobRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.Map;
import java.util.function.BiFunction;

import static com.stonewu.fusion.service.canvas.agent.CanvasAgentSchema.*;

@Configuration
@RequiredArgsConstructor
public class CanvasAgentToolsConfiguration {
    private final CanvasAgentQueries queries;
    private final CanvasAgentMutations mutations;
    private final CanvasAgentRunTools runs;
    private final CanvasAgentAwaitService waits;
    private final CanvasAgentMediaInspection inspection;
    private final CanvasAgentOutputBinding bindings;
    private final CanvasAgentLayout layouts;
    private final CanvasAgentAccess access;
    private final CanvasWorkspaceService workspace;
    private final CanvasUndoService undo;
    private final CanvasJson json;

    @Bean
    ToolExecutor getCanvasMcpOperationsTool(CanvasAgentMcp mcp) {
        return tool("get_canvas_mcp_operations", "查看已登记 MCP 操作",
                "分页读取当前账号明确登记的画布 MCP 操作，每页 20 项。配置 mcpTool 节点前使用真实契约 ID 和版本；传 contractId 分页查看输入输出、副作用和超时约定。不会返回连接凭据，也不能由智能体自行登记新权限。",
                schema(Map.of("page", integer("目录页码，默认 0"), "contractId", string("读取详细契约的真实 ID"),
                        "offset", integer("契约正文起始字符，默认 0"), "limit",
                        integer("契约正文长度，默认 8000，最多 32000"))), true, mcp::catalog);
    }

    @Bean
    ToolExecutor getCanvasMcpCallTool(CanvasAgentMcp mcp) {
        return tool("get_canvas_mcp_call", "查看 MCP 原调用",
                "分页读取本画布 MCP 节点的实际调用状态、输入、已接收结果与人工核实记录。未知调用必须由用户核实，不得通过新建相同节点规避待核实状态。",
                schema(Map.of("canvasRunId", string("画布运行 ID"), "nodeRunId", string("MCP 节点运行 ID"),
                                "offset", integer("正文起始字符，默认 0"), "limit", integer("正文长度，默认 8000，最多 32000")),
                        "canvasRunId", "nodeRunId"), true, mcp::execution);
    }

    @Bean
    ToolExecutor importCanvasSourcesTool(CanvasAgentImports imports) {
        return tool("import_canvas_sources", "导入素材来源",
                "将已查询到的业务或生成条目媒体导入托管素材库。sources 每项含 title 与 source={type,id,field,version}，最多 50 项。"
                        + "source.type 使用 project/asset/assetItem/shot/storyboardEpisode/imageItem/videoItem，field 使用真实的媒体字段名。"
                        + "不接受任意 URL；先等待完成，再放入画布。",
                schema(Map.of("sources", array(object("{title,source:{type,id,field,version}}"), 50)), "sources"),
                false,
                imports::submit);
    }

    @Bean
    ToolExecutor getCanvasImportTool(CanvasAgentImports imports) {
        return tool("get_canvas_import", "查看素材导入",
                "读取此画布导入批次的逐项状态、失败原因、真实媒体引用及放入回执。",
                schema(Map.of("importId", string("导入任务 ID")), "importId"), true,
                (input, context) -> imports.get(text(input, "importId"), context));
    }

    @Bean
    ToolExecutor awaitCanvasImportTool(CanvasAgentImports imports) {
        return new CanvasImportAwaitTool(imports, json);
    }

    @Bean
    ToolExecutor controlCanvasImportTool(CanvasAgentImports imports) {
        return tool("control_canvas_import", "管理素材导入",
                "取消本画布的导入批次，或在用户要求重试时仅重试明确失败/取消的条目。重试须携带当前版本和条目编号，不重试已完成项。",
                schema(Map.of("importId", string("导入任务 ID"), "action", choice("动作", "cancel", "retry"),
                        "expectedRevision", integer("重试时的导入版本"), "indices",
                        array(integer("失败或取消的条目编号"), 50)), "importId", "action"), false,
                imports::control);
    }

    @Bean
    ToolExecutor placeCanvasImportTool(CanvasAgentImports imports) {
        return tool("place_canvas_import", "放入已导入素材",
                "把此导入批次已完成的素材作为节点放入画布，节点和导入回执一起保存，重复调用不会重建用户已移除的节点。",
                schema(Map.of("importId", string("导入任务 ID"), "x", number("落点横坐标"), "y", number("落点纵坐标")),
                        "importId", "x", "y"), false,
                (input, context) -> imports.place(text(input, "importId"), input.path("x").asDouble(),
                        input.path("y").asDouble(), context));
    }

    @Bean
    ToolExecutor queryCanvasLibraryTool(CanvasAgentLibrary library) {
        return tool("query_canvas_library", "查找项目素材", "分页读取当前画布作用域中的真实剧本、分镜、资产和托管媒体，包括尚未放入画布的内容。"
                        + "project 返回项目、剧本和分镜容器；children 展开父对象下一层；assets 搜索主资产；media 列出托管媒体。每页 40 项，page 从 0 开始。"
                        + "imageHistory/videoHistory 列出本画布项目中的个人生成历史；传 parentId=任务数字 id 展开结果及可导入的 sources。"
                        + "可用返回的 nodeKind/resource/config 创建资源节点；需要正文时调用 get_canvas_library_resource。",
                schema(Map.of("category",
                        choice("目录", "project", "children", "assets", "media", "imageHistory", "videoHistory"),
                        "parentType", string("展开时的父对象类型"),
                        "parentId", integer("父对象编号"), "query", string("资产名称搜索词"), "page",
                        integer("页码，默认 0")), "category"), true,
                (input, context) -> library.query(text(input, "category"), text(input, "parentType"),
                        input.hasNonNull("parentId") ? input.path("parentId").longValue() : null,
                        text(input, "query"), integer(input, "page", 0), context));
    }

    @Bean
    ToolExecutor getCanvasLibraryResourceTool(CanvasAgentLibrary library) {
        return tool("get_canvas_library_resource", "读取项目素材",
                "按刚查询的真实业务类型和编号读取当前版本正文，仅限本画布的项目。长 JSON 按字符分页。",
                schema(Map.of("type", string("业务资源类型"), "id", integer("业务资源编号"), "offset",
                        integer("正文起始字符，默认 0"),
                        "limit", integer("正文长度，默认 8000，最多 32000")), "type", "id"), true,
                (input, context) -> library.details(text(input, "type"), input.path("id").longValue(),
                        integer(input, "offset", 0), integer(input, "limit", 8000), context));
    }

    @Bean
    ToolExecutor getCanvasModelsTool(CanvasAgentModels models) {
        return tool("get_canvas_models", "查看可用模型", "分页列出可用于画布的已启用模型及音频服务，每页 40 项，page 从 0 开始。"
                        + "传 modelId 读取完整能力和已发布工作流版本，details 为按字符分页的 JSON。配置节点前使用真实 ID 核对参数，不返回凭据。",
                schema(Map.of("kind", choice("模型类别", "text", "image", "video", "comfyui"), "page",
                        integer("目录页码，默认 0"),
                        "modelId", integer("完整能力的模型 ID"), "offset", integer("能力 JSON 的起始字符，默认 0"),
                        "limit", integer("能力 JSON 长度，默认 8000，最多 32000"))), true, models::query);
    }

    @Bean
    ToolExecutor awaitGenerationTaskTool(CanvasQueryService canvases, GenerationJobRepository jobs,
                                         GetGenerationTaskToolExecutor query) {
        return new CanvasGenerationAwaitTool(canvases, jobs, query, json);
    }

    @Bean
    ToolExecutor getCanvasContextTool() {
        return tool("get_canvas_context", "查看画布",
                "读取本轮绑定画布的概要、权限、选中范围和能力目录。通过 kinds 按需读取至多 8 种节点的端口及参数，不能指定其他画布。",
                schema(Map.of("kinds", array(CanvasAgentGraphSchema.kind(), 8))), true,
                (input, context) -> queries.context(strings(input, "kinds"), context));
    }

    @Bean
    ToolExecutor queryCanvasNodesTool() {
        return tool("query_canvas_nodes", "查找节点",
                "按类型、名称或相邻节点分页查询当前画布；返回真实节点 ID 与当前版本。默认每页 40 个，修改前按需读取详情。",
                schema(Map.of("kind", CanvasAgentGraphSchema.kind(), "query", string("标题包含的文本"), "adjacentTo",
                        string("相邻节点 ID"),
                        "offset", integer("从第几个节点开始，默认 0"), "limit", integer("每页数量，默认 40，最多 100"))),
                true,
                (input, context) -> queries.query(text(input, "kind"), text(input, "query"), text(input, "adjacentTo"),
                        integer(input, "offset", 0), integer(input, "limit", 40), context));
    }

    @Bean
    ToolExecutor getCanvasNodeDetailsTool() {
        return tool("get_canvas_node_details", "读取节点内容",
                "读取节点、端口、连线和正式业务对象的当前版本。configuration 和 content 均为按字符分页的 JSON，默认 8000 字符。连线每页 100 条；输入 URL 文本不表示模型看到了媒体。",
                schema(Map.of("nodeId", string("真实节点 ID"), "offset", integer("正文起始字符，默认 0"), "limit",
                        integer("正文字数，默认 8000，最多 32000"),
                        "edgeOffset", integer("连线起始条目，默认 0")), "nodeId"), true,
                (input, context) -> queries.details(text(input, "nodeId"), integer(input, "offset", 0),
                        integer(input, "limit", 8000), integer(input, "edgeOffset", 0), context));
    }

    @Bean
    ToolExecutor getCanvasOutputTool() {
        return tool("get_canvas_output", "读取运行产物",
                "分页读取某个真实运行产物的完整内容，必须使用刚查询到的 nodeRunId 和 outputId；正文为 JSON。",
                schema(Map.of("canvasRunId", string("画布运行 ID"), "nodeRunId", string("节点运行 ID"), "outputId",
                                string("产物 ID"),
                                "offset", integer("正文起始字符，默认 0"), "limit", integer("正文长度，默认 8000，最多 32000")),
                        "canvasRunId", "nodeRunId", "outputId"), true,
                (input, context) -> runs.output(text(input, "canvasRunId"), text(input, "nodeRunId"),
                        text(input, "outputId"),
                        integer(input, "offset", 0), integer(input, "limit", 8000), context));
    }

    @Bean
    ToolExecutor getCanvasFixedOutputTool() {
        return tool("get_canvas_fixed_output", "读取连线固定结果",
                "按连线 ID 分页读取实际固定结果，包括画布副本保留的快照。与当前画布及底层资源权限一致；不会执行或重跑上游节点。",
                schema(Map.of("edgeId", string("当前画布的固定结果连线 ID"), "offset", integer("起始字符，默认 0"),
                        "limit", integer("正文长度，默认 8000，最多 32000")), "edgeId"), true,
                (input, context) -> runs.fixedOutput(text(input, "edgeId"), integer(input, "offset", 0),
                        integer(input, "limit", 8000), context));
    }

    @Bean
    ToolExecutor getCanvasWritebackTool() {
        return tool("get_canvas_writeback", "读取写回提案",
                "分页读取已固定的业务写回提案，包含原值、新值、版本和影响范围。确认前核对 hash、危险重置说明与目标对象。",
                schema(Map.of("canvasRunId", string("画布运行 ID"), "nodeRunId", string("写回节点运行 ID"), "offset",
                        integer("正文起始字符，默认 0"),
                        "limit", integer("正文长度，默认 8000，最多 32000")), "canvasRunId", "nodeRunId"), true,
                (input, context) -> runs.writeback(text(input, "canvasRunId"), text(input, "nodeRunId"),
                        integer(input, "offset", 0), integer(input, "limit", 8000), context));
    }

    @Bean
    ToolExecutor queryCanvasOutputsTool() {
        return tool("query_canvas_outputs", "查找运行产物",
                "分页列出节点运行某个端口的真实产物，包括候选输入；用真实 outputId 选择候选、观察媒体或放入画布。",
                schema(Map.of("canvasRunId", string("画布运行 ID"), "nodeRunId", string("节点运行 ID"), "port",
                                string("输出端口，候选输入使用 candidates"),
                                "offset", integer("起始条目，默认 0"), "limit", integer("每页数量，默认 20，最多 50")),
                        "canvasRunId", "nodeRunId", "port"), true,
                (input, context) -> runs.outputs(text(input, "canvasRunId"), text(input, "nodeRunId"),
                        text(input, "port"),
                        integer(input, "offset", 0), integer(input, "limit", 20), context));
    }

    @Bean
    ToolExecutor applyCanvasOperationsTool() {
        return tool("apply_canvas_operations", "修改画布",
                "原子应用 1 至 100 个 CanvasGraphOperation：canvas.node.create/patch/layout/bind/delete/assert、canvas.edge.create/replace/delete。"
                        + "create 的 node/edge.id 使用 $名称 别名，本批 nodeId/sourceNodeId/targetNodeId/parentId 可引用该别名；回执提供真实 ID。"
                        + "node 必须包含 kind、schemaVersion=1、title、layout、config、configVersion=0、layoutVersion=0；layout 含 x、y、locked、collapsed。"
                        + "patch/bind/assert/delete 必须携带 expectedConfigVersion，layout/delete 必须携带 expectedLayoutVersion，edge.replace/delete 必须携带 expectedVersion。"
                        + "patch 用 configPatch 修改参数；bind 用 resource={type,id,version}。删除节点仅移除视图，不删除业务数据。此工具不会开始生成。",
                schema(Map.of("operations", CanvasAgentGraphSchema.operations()), "operations"), false,
                (input, context) -> {
                    if (!(input.get("operations") instanceof ArrayNode values)) {
                        throw new BusinessException(400, "operations 必须为列表");
                    }
                    return mutations.apply(values, context);
                });
    }

    @Bean
    ToolExecutor editCanvasResourceTool() {
        return tool("edit_canvas_resource", "编辑业务内容",
                "通过统一业务命令修改画布上的剧本、场次、分镜或资产。patch/delete 使用节点绑定的真实业务对象及 expectedVersion；"
                        + "create 提供 resourceType、title、fields、x/y，新对象与画布节点在同一事务创建。改写原文或删除业务对象属于高风险操作。fields 必须使用业务协议字段，不接受用户身份。",
                schema(Map.ofEntries(Map.entry("action",
                                        choice("操作；replaceSource 会替换剧本原文并重置解析结果", "create", "patch", "delete",
                                                "replaceSource")), Map.entry("nodeId", string("修改或删除时的节点 ID")),
                                Map.entry("resourceType", string("新建的业务类型")), Map.entry("title", string("新节点名称")),
                                Map.entry("expectedVersion", integer("当前业务版本")),
                                Map.entry("expectedParentVersion", integer("资产子项需要的父级版本")),
                                Map.entry("fields", object("新建或修改的类型化业务字段")),
                                Map.entry("x", number("新节点横坐标，默认 0")), Map.entry("y", number("新节点纵坐标，默认 0"))),
                        "action"), false,
                (input, context) -> mutations.edit(text(input, "action"), text(input, "nodeId"),
                        text(input, "resourceType"), text(input, "title"),
                        optionalInteger(input, "expectedVersion"), optionalInteger(input, "expectedParentVersion"),
                        object(input, "fields"),
                        coordinate(input, "x"), coordinate(input, "y"), context));
    }

    @Bean
    ToolExecutor validateCanvasRunTool() {
        return tool("validate_canvas_run", "预览运行计划",
                "校验节点或子图、模型、媒体权限、输入版本和用量上限，只生成十分钟有效的冻结计划，不调用供应商。必须明确预算；仅使用节点已配置的模型。",
                schema(Map.of("expectedRevision", integer("当前画布版本"), "mode",
                                choice("运行范围", "SINGLE", "SELECTION", "DEPENDENCIES", "ALL", "STALE"),
                                "nodeIds", array(string("运行节点 ID"), 200), "reuse",
                                object("nodeId 到已有成功 nodeRunId 的显式复用映射"),
                                "budget", schema(Map.of("maxNodes", integer("最多节点数"), "maxImages", integer("最多图片数"),
                                                "maxVideoSeconds", integer("最多视频秒数"),
                                                "maxParallel", integer("最大同时执行数"), "maxBatchItems", integer("最多批量条目数")),
                                        "maxNodes", "maxImages", "maxVideoSeconds", "maxParallel", "maxBatchItems")),
                        "expectedRevision", "mode", "nodeIds", "budget"), true,
                (input, context) -> {
                    var budget = object(input, "budget");
                    if (budget == null) {
                        throw new BusinessException(400, "缺少运行预算");
                    }
                    return runs.plan(version(input, "expectedRevision"), text(input, "mode"), strings(input, "nodeIds"),
                            stringsMap(input, "reuse"),
                            new Budget(integer(budget, "maxNodes", -1), integer(budget, "maxImages", -1),
                                    integer(budget, "maxVideoSeconds", -1),
                                    integer(budget, "maxParallel", -1), integer(budget, "maxBatchItems", -1)), context);
                });
    }

    @Bean
    ToolExecutor runCanvasNodesTool() {
        return tool("run_canvas_nodes", "运行画布节点",
                "提交已校验且仍有效的运行计划，可能消耗模型额度。返回真实 canvasRunId；请使用 await_canvas_run 等待，禁止重复启动来确认结果。",
                schema(Map.of("planId", string("validate_canvas_run 返回的计划 ID"), "planHash",
                        string("同一计划的内容摘要")), "planId", "planHash"), false,
                (input, context) -> runs.start(text(input, "planId"), text(input, "planHash"), context));
    }

    @Bean
    ToolExecutor getCanvasRunTool() {
        return tool("get_canvas_run", "读取运行结果",
                "读取当前画布运行的真实状态、输出 ID、统计和需要处理的等待点。非终态时使用 await_canvas_run，避免循环查询。",
                schema(Map.of("canvasRunId", string("画布运行 ID"), "offset", integer("起始节点，默认 0"), "limit",
                        integer("节点数，默认 20，最多 50")), "canvasRunId"), true,
                (input, context) -> runs.get(text(input, "canvasRunId"), integer(input, "offset", 0),
                        integer(input, "limit", 20), context));
    }

    @Bean
    ToolExecutor reviewCanvasBatchTool() {
        return tool("review_canvas_batch", "处理批量结果", "处理已经等待确认的 BATCH 节点。RETRY_FAILED 仅显式重试终态失败项，产生新的执行用量；"
                        + "USE_SUCCEEDED 按原顺序汇总已有结果继续下游；FINISH 保留结果并结束批次。先读取每项状态和当前节点 revision，不能自动重试未知请求。",
                schema(Map.of("canvasRunId", string("画布运行 ID"), "nodeRunId", string("BATCH 节点运行 ID"),
                                "expectedRevision", integer("当前节点执行版本"),
                                "action", choice("处理方式", "RETRY_FAILED", "USE_SUCCEEDED", "FINISH"), "note",
                                string("处理说明")),
                        "canvasRunId", "nodeRunId", "expectedRevision", "action", "note"), false,
                (input, context) -> runs.reviewBatch(text(input, "canvasRunId"), text(input, "nodeRunId"),
                        version(input, "expectedRevision"), text(input, "action"), text(input, "note"), context));
    }

    @Bean
    ToolExecutor awaitCanvasRunTool() {
        return new CanvasAwaitTool(waits, json);
    }

    @Bean
    ToolExecutor inspectCanvasMediaTool() {
        return new CanvasInspectTool(inspection, json);
    }

    @Bean
    ToolExecutor bindCanvasOutputTool() {
        return tool("bind_canvas_output", "采用运行结果",
                "把当前画布运行的确定 outputId 写入目标节点的正式业务字段。binding 使用写回节点能力中的字段标识，如 shot:firstFrameImageUrl。"
                        + "必须携带当前业务版本；资产变体还需父级版本。写入剧本原文会重置解析结果。",
                schema(Map.of("canvasRunId", string("画布运行 ID"), "outputId", string("已发布的输出 ID"),
                                "targetNodeId", string("目标业务节点 ID"),
                                "binding", string("写回字段标识"), "expectedVersion", integer("目标业务版本"),
                                "expectedParentVersion", integer("资产父级版本")),
                        "canvasRunId", "outputId", "targetNodeId", "binding", "expectedVersion"), false,
                (input, context) -> bindings.bind(text(input, "canvasRunId"), text(input, "outputId"),
                        text(input, "targetNodeId"),
                        text(input, "binding"), integer(input, "expectedVersion", -1),
                        optionalInteger(input, "expectedParentVersion"), context));
    }

    @Bean
    ToolExecutor confirmCanvasWritebackTool() {
        return tool("confirm_canvas_writeback", "处理写回提案",
                "处理 WAITING_CONFIRMATION 写回节点。必须先读 get_canvas_run 的真实提案及影响对象，提交原 proposalHash 和 nodeRun revision；不能复用旧提案覆盖新版本。",
                schema(Map.of("canvasRunId", string("画布运行 ID"), "nodeRunId", string("写回执行节点 ID"),
                                "expectedRevision", integer("节点执行版本"),
                                "proposalHash", string("提案内容摘要"), "action", choice("处理方式", "APPROVE", "SKIP"), "note",
                                string("明确的处理说明"),
                                "acceptReset", bool("是否明确接受提案中列出的剧本重置")), "canvasRunId", "nodeRunId",
                        "expectedRevision", "proposalHash", "action", "note"), false,
                (input, context) -> bindings.confirm(text(input, "canvasRunId"), text(input, "nodeRunId"),
                        version(input, "expectedRevision"),
                        text(input, "proposalHash"), text(input, "action"), text(input, "note"),
                        input.path("acceptReset").asBoolean(), context));
    }

    @Bean
    ToolExecutor cancelCanvasRunTool() {
        return tool("cancel_canvas_run", "停止画布运行",
                "停止自己在当前画布发起的运行。远端任务可能仍在结束中；以返回的真实终态为准，不删除已完成产物。",
                schema(Map.of("canvasRunId", string("画布运行 ID")), "canvasRunId"), false,
                (input, context) -> runs.cancel(text(input, "canvasRunId"), context));
    }

    @Bean
    ToolExecutor selectCanvasCandidateTool() {
        return tool("select_canvas_candidate", "选择生成候选",
                "从 WAITING_SELECTION 节点的真实 candidates 中选择 outputId，继续下游。必须携带该 nodeRun 的当前 revision。",
                schema(Map.of("canvasRunId", string("画布运行 ID"), "nodeRunId", string("候选选择执行节点 ID"),
                                "expectedRevision", integer("执行节点 revision"),
                                "outputId", string("返回的候选输出 ID")), "canvasRunId", "nodeRunId", "expectedRevision",
                        "outputId"), false,
                (input, context) -> runs.select(text(input, "canvasRunId"), text(input, "nodeRunId"),
                        version(input, "expectedRevision"), text(input, "outputId"), context));
    }

    @Bean
    ToolExecutor focusCanvasNodesTool() {
        return tool("focus_canvas_nodes", "定位节点",
                "向发起会话发送可忽略的节点定位提示。只影响愿意跟随助手的本地视口，不修改其他用户视图。",
                schema(Map.of("nodeIds", array(string("节点 ID"), 200)), "nodeIds"), true, (input, context) -> {
                    var session = access.require(context);
                    var ids = strings(input, "nodeIds");
                    ids.forEach(session::node);
                    return Map.of("status", "success", "focus",
                            Map.of("canvasId", session.scope().canvasId(), "nodeIds", ids,
                                    "agentRunId", session.context().agentRunId(), "toolCallId",
                                    session.context().toolCallId()));
                });
    }

    @Bean
    ToolExecutor arrangeCanvasNodesTool() {
        return tool("arrange_canvas_nodes", "整理节点",
                "对指定节点按流程从左到右排列，优先把连续链路放在同一水平线上；支持对齐和横纵排列，保留锁定节点与分组关系。修改前须提供当前画布版本。",
                schema(Map.of("nodeIds", array(string("本次整理的节点 ID"), 200), "mode",
                        choice("排列方式", "arrange", "left", "top", "horizontal", "vertical"),
                        "expectedRevision", integer("当前画布版本")), "nodeIds", "mode", "expectedRevision"), false,
                (input, context) -> layouts.arrange(strings(input, "nodeIds"), text(input, "mode"),
                        version(input, "expectedRevision"), context));
    }

    @Bean
    ToolExecutor createCanvasCheckpointTool() {
        return tool("create_canvas_checkpoint", "保存命名版本",
                "记录当前画布与业务内容的命名版本。expectedRevision 必须与最新画布一致；后续撤销依然检查当前版本，不能覆盖其他人的修改。",
                schema(Map.of("title", string("版本名称"), "expectedRevision", integer("画布版本")), "title",
                        "expectedRevision"), false, (input, context) -> {
                    var session = access.require(context);
                    return Map.of("status", "success", "receipt",
                            workspace.checkpoint(session.scope().canvasId(), new CheckpointCreate(session.requestId(),
                                            session.id("checkpoint"), text(input, "title"), version(input, "expectedRevision")),
                                    session.context()));
                });
    }

    @Bean
    ToolExecutor undoCanvasCommandTool() {
        return tool("undo_canvas_command", "撤销画布修改",
                "有条件撤销指定画布图命令；后续有人改动时返回冲突。此操作不会撤销已执行的生成或正式业务字段修改。",
                schema(Map.of("commandId", string("需要撤销的真实画布图命令 ID")), "commandId"), false,
                (input, context) -> {
                    var session = access.require(context);
                    return Map.of("status", "success", "receipt", undo.undo(session.scope().canvasId(),
                            new CanvasUndoCommand(session.requestId(), text(input, "commandId")), session.context()));
                });
    }

    private CanvasAgentTool tool(String name, String label, String description, ObjectNode schema, boolean readOnly,
                                 BiFunction<ObjectNode, ToolExecutionContext, Object> handler) {
        return new CanvasAgentTool(name, label, description, schema, readOnly, json, handler);
    }

    private Integer optionalInteger(ObjectNode input, String name) {
        return input.hasNonNull(name) ? integer(input, name, 0) : null;
    }

    private double coordinate(ObjectNode input, String name) {
        if (!input.hasNonNull(name)) {
            return 0;
        }
        if (!input.get(name).isNumber() || !Double.isFinite(input.get(name).doubleValue())) {
            throw new BusinessException(400, name + " 必须为有限坐标");
        }
        return input.get(name).doubleValue();
    }
}
