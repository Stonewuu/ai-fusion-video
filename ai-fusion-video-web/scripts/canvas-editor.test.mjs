import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { createRequire } from "node:module";
import ts from "typescript";

const require = createRequire(import.meta.url);
const root = path.resolve(import.meta.dirname, "..");
function moduleLoader(mocks = {}) {
  const cache = new Map();
  function load(relative) {
    const file = path.resolve(root, relative);
    if (mocks[file]) return mocks[file];
    if (cache.has(file)) return cache.get(file).exports;
    const loadedModule = { exports: {} };
    cache.set(file, loadedModule);
    const compiled = ts.transpileModule(fs.readFileSync(file, "utf8"), {
      compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, esModuleInterop: true },
    }).outputText;
    const localRequire = (name) => {
      if (name.startsWith("@/")) return load(`${name.slice(2)}.ts`);
      if (name.startsWith(".")) return load(`${path.resolve(path.dirname(file), name)}.ts`);
      return require(name);
    };
    vm.runInThisContext(`(function(require,module,exports){${compiled}\n})`, { filename: file })(localRequire, loadedModule, loadedModule.exports);
    return loadedModule.exports;
  }
  return load;
}
const load = moduleLoader({
  [path.join(root, "lib/api/client.ts")]: { getApiBaseUrl: () => "http://test.invalid" },
  [path.join(root, "lib/api/ai-assistant.ts")]: { authenticatedFetch: () => { throw new Error("Unexpected network request"); } },
});
const graphTools = load("lib/canvas/graph.ts");
const layouts = load("lib/canvas/layout-actions.ts");
const drafts = load("lib/canvas/resource-draft.ts");
const streams = load("lib/canvas/event-stream.ts");
const clipboard = load("lib/canvas/clipboard.ts");
const placements = load("lib/canvas/placement.ts");
const { userFacingError } = load("lib/user-facing-error.ts");
const errorPolicy = load("lib/client-error-policy.ts");
const apiErrors = load("lib/api/api-error.ts");
const { usesExternalCreation, runUsageSummary } = load("lib/canvas/run-presentation.ts");
const { canvasToolPresentation } = load("lib/canvas/tool-presentation.ts");
const { historyText } = load("lib/canvas/history-presentation.ts");
const { canvasNodeMediaGroups } = load("lib/canvas/node-media.ts");
const { canvasInputCandidates, connectCanvasInput } = load("lib/canvas/input-bindings.ts");
const { canvasRecoveryPlan } = load("lib/canvas/run-recovery.ts");
const { canvasAgentExecutionFinished } = load("lib/canvas/agent-execution.ts");
const { validateCanvasConfig } = load("lib/canvas/config.ts");
const { compactAgentOverrides, effectiveAgentValues } = load("lib/api/agent-definitions.ts");
const nodeDraftTools = load("lib/canvas/node-draft.ts");
const fieldPresentation = load("lib/canvas/field-presentation.ts");
const nodeExecution = load("lib/canvas/node-execution.ts");
const { createCanvasNodeRunStore } = load("lib/canvas/node-run-store.ts");

test("direct node runs traverse execution dependencies and reuse only current successful ancestors", () => {
  const graph = { nodes: [], edges: [
    { kind: "execution", sourceNodeId: "text", targetNodeId: "image", output: null },
    { kind: "execution", sourceNodeId: "image", targetNodeId: "video", output: null },
    { kind: "execution", sourceNodeId: "fixed", targetNodeId: "video", output: { outputId: "pinned" } },
    { kind: "reference", sourceNodeId: "reference", targetNodeId: "video" },
  ] };
  const activity = (nodeId, status = "SUCCEEDED", derivedState = "CURRENT", contentUnavailable = false) => ({ node: { nodeId, id: nodeId + "-run", status, derivedState, contentUnavailable } });
  assert.deepEqual(nodeExecution.canvasNodeDependencies(graph, "video"), ["video", "image", "text"]);
  assert.deepEqual(nodeExecution.reusableNodeDependencies(graph, "video", [activity("text"), activity("image"), activity("video"), activity("fixed")]), { text: "text-run", image: "image-run" });
  assert.deepEqual(nodeExecution.reusableNodeDependencies(graph, "video", [activity("text", "SUCCEEDED", "STALE"), activity("image", "SUCCEEDED", "CURRENT", true)]), {});
  assert.equal(nodeExecution.canRunCanvasNode({ executor: "resource" }), false);
  assert.equal(nodeExecution.canRunCanvasNode({ executor: "none" }), false);
  assert.equal(nodeExecution.canRunCanvasNode({ executor: "image" }), true);
  assert.equal(nodeExecution.canRunCanvasNode({ executor: "agent" }), true);
});

test("node activity refresh retains identities without mixing other nodes or old selected run history", () => {
  const store = createCanvasNodeRunStore();
  const activity = (nodeId, runId, status) => ({ runId, runStatus: status, canControl: true, children: [],
    node: { id: runId + ":" + nodeId, nodeId, parentNodeRunId: null, revision: 1, status, derivedState: "CURRENT", contentUnavailable: false } });
  const first = activity("a", "run-a", "RUNNING"), second = activity("b", "run-b", "SUCCEEDED");
  store.getState().updateActivities([first, second]);
  const before = store.getState();
  store.getState().updateActivities(JSON.parse(JSON.stringify([first, second])));
  assert.equal(store.getState().activities, before.activities);
  assert.equal(store.getState().nodes, before.nodes);
  store.getState().updateActivities([{ ...first, node: { ...first.node, revision: 2, status: "SUCCEEDED" } }, second]);
  assert.equal(store.getState().activities.get("b"), before.activities.get("b"));
  assert.equal(store.getState().nodes.get("a").status, "SUCCEEDED");
  store.getState().updateActivities([]);
  assert.equal(store.getState().activities.size, 0);
  assert.equal(store.getState().nodes.size, 0);
});

test("node progress counts real completed work and preserves the service failure reason", () => {
  const result = nodeExecution.generationNodeProgress({ items: [
    { state: "SUCCEEDED", phase: "FINISHED" }, { state: "FAILED", phase: "FINISHED", errorMessage: "video queue is full" },
    { state: "RUNNING", phase: "POLLING" },
  ] });
  assert.deepEqual(result, { completed: 2, succeeded: 1, total: 3, label: "正在生成", error: "video queue is full" });
  assert.deepEqual(nodeExecution.batchNodeProgress({ inputs: { values: { items: [1,2,3,4] } } }, [{ status: "SUCCEEDED" }, { status: "FAILED" }, { status: "RUNNING" }]), { completed: 2, total: 4 });
  const active = { node: { nodeId: "busy", status: "WAITING_SELECTION" } };
  assert.equal(nodeExecution.activePlanNode({ nodes: [{ nodeId: "busy", action: "EXECUTE" }] }, [active]), active);
  assert.equal(nodeExecution.activePlanNode({ nodes: [{ nodeId: "busy", action: "REUSE" }] }, [active]), undefined);
});

test("node editor derives choices and duration limits from the selected model without changing saved values", () => {
  const field = { type: "integer", minimum: 1, maximum: 3600, enumValues: [] };
  const model = { parameters: { minDuration: 4, maxDuration: 12, supportedAspectRatios: ["16:9", "9:16"], supportedResolutions: ["720P"] } };
  const config = { duration: 20, ratio: "4:3", resolution: "1080P", generateAudio: false };
  const before = JSON.stringify(config);
  assert.deepEqual(fieldPresentation.configNumberBounds("duration", field, model), { min: 4, max: 12 });
  const ratios = fieldPresentation.configFieldChoices("videoGeneration", "ratio", field, model);
  assert.deepEqual(ratios.map(x => x.value), ["16:9", "9:16"]);
  assert.match(fieldPresentation.configFieldError("ratio", field, config.ratio, ratios, model), /支持范围/);
  assert.match(fieldPresentation.configFieldError("duration", field, config.duration, [], model), /4–12/);
  assert.equal(JSON.stringify(config), before);
  assert.equal(fieldPresentation.configFieldError("duration", field, undefined, [], model), null);
});

test("image size tiers and output choices retain their actual protocol values", () => {
  const field = { enumValues: [] };
  const model = { parameters: { supportedSizes: { "1K": { "16:9": "1312x736" }, "2K": { "16:9": "2624x1472" } } } };
  assert.deepEqual(fieldPresentation.configFieldChoices("imageGeneration", "resolution", field, model).map(x => x.value), ["1K", "2K"]);
  assert.equal(fieldPresentation.configuredImageSize({ resolution: "1K", ratio: "16:9" }, model), "1312x736");
  assert.equal(fieldPresentation.configuredImageSize({ resolution: "8K", ratio: "16:9" }, model), null);
  assert.equal(fieldPresentation.configFieldChoices("videoCompose", "resolution", field)[1].value, "1920x1080");
  assert.deepEqual(fieldPresentation.configFieldChoices("textCombine", "separator", field).map(x => x.value), ["\n\n", "\n", " ", ""]);
});

test("unsupported camera option is warned only for explicit enable, defaults and disable remain valid", () => {
  const field = { type: "boolean", enumValues: [] };
  const model = { parameters: { supportCameraFixed: false } };
  assert.match(fieldPresentation.configFieldError("cameraFixed", field, true, [], model), /不支持固定镜头/);
  assert.equal(fieldPresentation.configFieldError("cameraFixed", field, false, [], model), null);
  assert.equal(fieldPresentation.configFieldError("cameraFixed", field, undefined, [], model), null);
});

test("identical first and last frame files remain independently replaceable", () => {
  const resource = { header: { resourceType: "shot", resourceId: 1, projectId: 1, version: 1 }, data: { firstFrameImageUrl: "media://frame", lastFrameImageUrl: "media://frame" } };
  const groups = canvasNodeMediaGroups({ kind: "shot", title: "镜头", config: {}, resource: { type: "shot", id: 1, version: 1 } }, resource, undefined);
  assert.equal(groups[0].items.length, 2);
  assert.deepEqual(groups[0].items.map(item => item.field), ["firstFrameImageUrl", "lastFrameImageUrl"]);
});

test("a media source previews its current file after replacement instead of a previous run snapshot", () => {
  const groups = canvasNodeMediaGroups({ kind: "image", title: "图片", config: { resourceId: "new-file" } }, null,
    { inputs: { values: {} }, outputs: { values: { media: [{ mediaId: "old-file", type: "image" }] } } });
  assert.equal(groups.length, 1);
  assert.equal(groups[0].label, "内容预览");
  assert.equal(groups[0].items[0].source, "media://new-file");
});

test("Skill arrays and nested operation drafts survive reload without phantom edits or conflicts", () => {
  const base = { id: "node", title: "分镜", config: { skillMode: "CUSTOM", skillNames: ["continuity"], operationConfig: { skillNames: ["review"] } } };
  const restored = JSON.parse(JSON.stringify({ base, title: base.title, config: base.config }));
  assert.equal(nodeDraftTools.isCanvasConfig(restored.config), true);
  assert.deepEqual(nodeDraftTools.changedNodeFields(restored), []);
  restored.config.skillNames = [];
  assert.deepEqual(nodeDraftTools.changedNodeFields(restored), ["skillNames"]);
  const remote = JSON.parse(JSON.stringify(base));
  remote.config.operationConfig.skillNames = ["different"];
  assert.deepEqual(nodeDraftTools.nodeDraftConflicts(restored, remote), []);
  assert.deepEqual(nodeDraftTools.rebaseNodeDraft(restored, remote, []).config, { skillMode: "CUSTOM", skillNames: [], operationConfig: { skillNames: ["different"] } });
  assert.equal(nodeDraftTools.isCanvasConfig({ skillNames: [null] }), false);
});

test("agent overrides distinguish inheritance from disabled and empty selections", () => {
  const defaults = { name: "内置", toolsEnabled: true, toolNames: ["read"], defaultSkills: ["review"], maxIters: 80 };
  const draft = { name: null, toolsEnabled: false, toolNames: [], defaultSkills: [], maxIters: 0 };
  assert.deepEqual(compactAgentOverrides(draft), { toolsEnabled: false, toolNames: [], defaultSkills: [], maxIters: 0 });
  assert.deepEqual(effectiveAgentValues(defaults, draft), { name: "内置", toolsEnabled: false, toolNames: [], defaultSkills: [], maxIters: 0 });
  assert.deepEqual(effectiveAgentValues(defaults, {}), defaults);
  assert.equal(defaults.toolsEnabled, true);
});

test("Skill selections accept an explicit empty list and reject duplicate, excessive or invalid entries", () => {
  const capability = { configFields: { skillNames: { type: "string_array", required: false, minimum: 0, maximum: 8, maxLength: 100, enumValues: [] } } };
  for (const value of [[], ["review"], ["review", "continuity"]]) assert.equal(validateCanvasConfig(capability, { skillNames: value }), null);
  for (const value of ["review", [""], ["review", "review"], [1], Array.from({ length: 9 }, (_, index) => `skill-${index}`)])
    assert.match(validateCanvasConfig(capability, { skillNames: value }), /选择无效/);
});
const node = (id, kind = "text", config = {}) => ({ id, kind, schemaVersion: 1, title: id, config, resource: null, configVersion: 0, layoutVersion: 0,
  layout: { x: 0, y: 0, width: 300, height: null, parentId: null, locked: false, collapsed: false } });
const port = (id, type, collection = false) => ({ id, label: id, valueType: type, valueTypeField: null, collection, required: false, maxConnections: collection ? 8 : 1 });
const capability = (kind, inputs, outputs = [], fields = {}) => ({ kind, label: kind, category: "test", inputs, outputs, configFields: fields, requiresProject: false, schemaVersion: 1, executor: "test", resourceType: null });
const edge = (id, sourceNodeId, sourcePort, targetNodeId, targetPort) => ({ id, sourceNodeId, sourcePort, targetNodeId, targetPort, kind: "execution", inputIndex: 0, output: null, version: 0 });

test("assets connect directly to generation as content while image inputs stay typed", () => {
  const context = { ...port("context", null, true), acceptedTypes: ["text", "asset", "assetItem", "shot"] };
  const asset = node("asset", "assetItem"), video = node("video", "videoGeneration", { prompt: "让角色走近镜头" });
  const image = node("image", "image");
  const catalog = [capability("assetItem", [], [port("assetItem", "assetItem"), port("image", "image")]),
    capability("videoGeneration", [port("prompt", "text"), context, port("firstFrame", "image")], [port("videos", "video", true)]),
    capability("image", [], [port("media", "image")])];
  const graph = { nodes: [asset, video, image], edges: [], projectId: null };
  assert.equal(graphTools.validateConnection(graph, catalog, "asset", "assetItem", "video", "context"), null);
  assert.match(graphTools.validateConnection(graph, catalog, "image", "media", "video", "context"), /不兼容/);
  assert.equal(graphTools.compatibleNodes(asset, catalog[0].outputs[0], catalog, null).some(item => item.capability.kind === "videoGeneration" && item.port.id === "context"), true);
  const options = canvasInputCandidates(graph, catalog, video, context);
  assert.equal(options.length, 1);
  assert.equal(options[0].node.id, "asset");
  const operations = connectCanvasInput(graph, catalog, "video", "context", "asset", "assetItem", "new-edge");
  assert.equal(operations.some(operation => operation.type === "canvas.node.patch"), false);
  assert.equal(operations.at(-1).edge.targetPort, "context");
  assert.equal(video.config.prompt, "让角色走近镜头");
});

test("replacing a scalar input is one versioned operation and never deletes the previous source node", () => {
  const target = node("target", "textGeneration", { prompt: "手填的内容" });
  const old = edge("old-edge", "old", "text", "target", "prompt");
  old.version = 3;
  const graph = { nodes: [node("old"), node("new"), target], edges: [old] };
  const catalog = [capability("text", [], [port("text", "text")]), capability("textGeneration", [port("prompt", "text")])];
  const operations = connectCanvasInput(graph, catalog, "target", "prompt", "new", "text", "unused-id");
  assert.deepEqual(operations.find(operation => operation.type === "canvas.node.patch").configPatch, { prompt: null });
  assert.equal(operations.at(-1).type, "canvas.edge.replace");
  assert.equal(operations.at(-1).edge.id, "old-edge");
  assert.equal(operations.at(-1).expectedVersion, 3);
  assert.equal(operations.some(operation => operation.type === "canvas.node.delete"), false);
});

test("content-only assets never appear as an image sent to the model", () => {
  const asset = { id: "asset", type: "assetItem", mediaId: null, resource: { type: "assetItem", id: 1, version: 0 }, data: { name: "角色", imageUrl: "media://character" } };
  const run = { contentUnavailable: false, inputs: { values: { context: [asset] } }, outputs: { values: {} } };
  assert.deepEqual(canvasNodeMediaGroups(node("video", "videoGeneration"), null, run), []);
  run.inputs.values.firstFrame = [{ type: "image", mediaId: "chosen-frame", resource: null, data: null }];
  assert.deepEqual(canvasNodeMediaGroups(node("video", "videoGeneration"), null, run)[0].items.map(item => item.source), ["media://chosen-frame"]);
});

test("recovery reuses current successful outputs and never resends unresolved submissions", () => {
  const item = (id, status, derivedState = "CURRENT") => ({ id: `run-${id}`, nodeId: id, status, derivedState, parentNodeRunId: null, contentUnavailable: false });
  const run = { canControl: true, status: "FAILED", nodes: [item("ready", "SUCCEEDED"), item("changed", "SUCCEEDED", "STALE"), item("failed", "FAILED"), item("next", "BLOCKED")] };
  assert.deepEqual(canvasRecoveryPlan(run), { mode: "DEPENDENCIES", nodeIds: ["failed", "next"], reuse: { ready: "run-ready" } });
  assert.equal(canvasRecoveryPlan({ ...run, status: "UNKNOWN_SUBMISSION" }), null);
  assert.equal(canvasRecoveryPlan({ ...run, canControl: false }), null);
});

test("completed agent text does not hide a generation that is still running or needs recovery", () => {
  const execution = { unresolvedModelCalls: 0, tasks: [{ status: "COMPLETED" }], generations: [{ state: "UNKNOWN_SUBMISSION" }] };
  for (const state of ["UNKNOWN_SUBMISSION", "NEEDS_REVIEW", "RUNNING", "QUEUED"]) {
    assert.equal(canvasAgentExecutionFinished({ ...execution, generations: [{ state }] }), false);
  }
  assert.equal(canvasAgentExecutionFinished({ ...execution, generations: [{ state: "SUCCEEDED" }] }), true);
  assert.equal(canvasAgentExecutionFinished({ ...execution, generations: [], unresolvedModelCalls: 1 }), false);
});

test("error presentation removes stack traces, credentials and implementation notices without hiding validation", () => {
  for (const message of ["java.sql.SQLException: secret-test-value", "schemaVersion 未实现", "错误 commandId: internal-test-value",
    "本地文件 D:/private/secret-test-value.txt 无法读取", "失败 api_key=secret-test-value", "返回 <!doctype html>secret-test-value</html>"]) {
    assert.doesNotMatch(userFacingError(message), /secret-test-value|internal-test-value|java\.sql|schemaVersion|commandId|api_key|doctype/);
  }
  assert.match(userFacingError("此节点执行器尚未接通"), /当前不可用/);
  assert.equal(userFacingError("当前模型最多支持 2 张参考图，请移除多余图片。"), "当前模型最多支持 2 张参考图，请移除多余图片。");
  assert.match(userFacingError("Network Error"), /连接服务/);
  assert.match(userFacingError("timeout of 10000ms exceeded"), /避免重复提交/);
  assert.equal(userFacingError('Error: {"status":"error","code":400,"message":"请选择图片模型","errorCode":"MODEL_REQUIRED"}'), "请选择图片模型");
  assert.match(userFacingError('Error: {"status":"error","message":"不支持的画布节点类型: comment"}'), /助手未能完成/);
  assert.doesNotMatch(userFacingError('Error: {"message":"失败",'), /Error|message|\{/);
  assert.doesNotMatch(userFacingError('Error: {"message":"图操作结构或字段无效，请按节点协议提交并携带预期版本"}'), /协议|预期版本|字段/);
  assert.equal(userFacingError("新建节点或连线必须使用 $名称 作为本批别名"), "助手未能完成这一步，请补充要求后重新尝试。");
  assert.equal(userFacingError("当前模型不支持 image 的 url 输入"), "当前模型无法读取这份图片，请选择支持图片输入的模型后继续。");
});

test("model errors explain the local action without sending the creator to settings", () => {
  assert.match(userFacingError('HTTP transport error during streaming: HTTP request failed with status 500 | Response body: {"error":{"message":"Expected canvas tool is unavailable"}}'), /所选模型.*本节点/);
  assert.match(userFacingError("HTTP request failed with status 503"), /503.*调用已结束/);
  assert.match(userFacingError("HTTP request failed with status 429"), /限流/);
  assert.doesNotMatch(userFacingError("原模型调用仍需核实，请先在执行核实中确认远端已结束"), /系统设置|前往/);
});

test("unconfirmed provider rejection keeps the review guidance without exposing gateway diagnostics", () => {
  const message = userFacingError("请求结果未确认，不能自动重发：agnes_25 视频任务提交失败: HTTP 503 video queue is full, please retry later (request id: provider-request-private)");
  assert.match(message, /视频服务繁忙.*受理状态尚未确认.*核实.*避免重复生成/);
  assert.doesNotMatch(message, /agnes_25|HTTP|provider-request-private|queue|已结束|重新运行/);
  assert.match(userFacingError("请求结果未确认：HTTP request failed with status 503"), /受理状态尚未确认.*避免重复提交/);
});

test("explicit provider failure preserves the actual reason and does not ask for review", () => {
  for (const raw of [
    "生成失败（HTTP 503）：video queue is full, please retry later (request id: provider-private)",
    "HTTP transport error during streaming: 生成失败（HTTP 503）：video queue is full, please retry later",
    "Agnes 2.5 视频任务提交失败: HTTP 503 video queue is full, please retry later (request id: provider-private)",
  ]) {
    const message = userFacingError(raw);
    assert.match(message, /生成失败（HTTP 503）.*供应商队列已满.*video queue is full, please retry later/);
    assert.doesNotMatch(message, /核实|尚未确认|provider-private/);
  }
  assert.equal(userFacingError("生成失败（HTTP 400）：Unsupported aspect ratio 21:9"), "生成失败（HTTP 400）：Unsupported aspect ratio 21:9");
  assert.match(userFacingError("生成失败（HTTP 401）：invalid api key"), /供应商拒绝访问.*invalid api key/);
  assert.match(userFacingError("生成失败（HTTP 402）：credit balance too low"), /供应商余额不足.*credit balance too low/);
});

test("node previews show asset inputs and every generated image or video without requiring storyboard nodes", () => {
  const image = { id: "image-1", type: "image", mediaId: "image", resource: null, data: null };
  const video = { id: "video-1", type: "video", mediaId: "video", resource: null, data: null };
  const asset = { id: "asset-1", type: "assetItem", mediaId: null, resource: { type: "assetItem", id: 1, version: 1 }, data: { imageUrl: "media://character" } };
  const run = { contentUnavailable: false, inputs: { values: { referenceImages: [image], assets: [asset] } }, outputs: { values: { videos: [video, { ...video, id: "video-2", mediaId: "video-two" }] } } };
  const groups = canvasNodeMediaGroups(node("generate", "videoGeneration"), null, run);
  assert.deepEqual(groups.map(group => group.label), ["输入参考", "运行结果"]);
  assert.deepEqual(groups[0].items.map(item => item.source), ["media://image", "media://character"]);
  assert.deepEqual(groups[1].items.map(item => item.source), ["media://video", "media://video-two"]);
  assert.deepEqual(canvasNodeMediaGroups(node("generate", "videoGeneration"), null, JSON.parse(JSON.stringify(run))), groups);
});

test("node previews include finished batch items, deduplicate aggregate results and remove inaccessible outputs", () => {
  const value = (id) => ({ id, type: "image", mediaId: id, data: null, resource: null });
  const run = { contentUnavailable: false, inputs: null, outputs: { values: { images: [value("one")] } } };
  const children = [
    { contentUnavailable: false, outputs: { values: { images: [value("one"), value("two")] } } },
    { contentUnavailable: true, outputs: { values: { images: [value("private")] } } },
  ];
  const groups = canvasNodeMediaGroups(node("batch", "foreach"), null, run, children);
  assert.deepEqual(groups[0].items.map(item => item.source), ["media://one", "media://two"]);
  assert.deepEqual(canvasNodeMediaGroups(node("batch", "foreach"), null, { ...run, contentUnavailable: true }, children), []);
});

test("a shot can preview both frames and its video and no image input is invented for text-to-video", () => {
  const resource = { data: { firstFrameImageUrl: "media://first", lastFrameImageUrl: "media://last", videoUrl: "media://video" } };
  const groups = canvasNodeMediaGroups(node("shot", "shot"), resource, undefined);
  assert.deepEqual(new Set(groups[0].items.map(item => item.source)), new Set(["media://first", "media://last", "media://video"]));
  const run = { contentUnavailable: false, inputs: { values: { prompt: [{ type: "text", text: "a scene", mediaId: null }] } }, outputs: { values: {} } };
  assert.deepEqual(canvasNodeMediaGroups(node("video", "videoGeneration"), null, run), []);
});

test("friendly API errors retain authorization status and do not turn cancelled control flow into an alert", () => {
  const original = { isAxiosError: true, response: { status: 403, data: { msg: "java.lang.IllegalStateException: internal-test-value" } } };
  const presented = apiErrors.toApiError(original);
  assert.doesNotMatch(presented.message, /java|internal-test-value/);
  assert.equal(presented.cause, original);
  assert.equal(apiErrors.getApiErrorStatus(presented), 403);
  const aborted = new Error("cancelled"); aborted.name = "AbortError";
  assert.equal(errorPolicy.isIgnoredClientError(aborted), true);
  assert.equal(errorPolicy.isIgnoredClientError(new Error("Network Error")), false);
});

test("the run review distinguishes existing content, reused generations and paid batch work", () => {
  const plan = (kind, action = "EXECUTE") => ({ nodes: [{ nodeId: "n", kind, action }] });
  assert.equal(usesExternalCreation(plan("text"), []), false);
  assert.equal(usesExternalCreation(plan("textGeneration"), []), true);
  assert.equal(usesExternalCreation(plan("videoGeneration", "REUSE"), []), false);
  assert.equal(usesExternalCreation(plan("foreach"), [node("n", "foreach", { operationKind: "imageGeneration" })]), true);
  assert.equal(usesExternalCreation(plan("foreach"), [node("n", "foreach", { operationKind: "imageTransform" })]), false);
  assert.equal(usesExternalCreation(plan("mcpTool"), []), true);
  assert.equal(runUsageSummary({ nodes: 1, images: 0, videoSeconds: 0 }), "最多 1 次处理");
});

test("canvas tool summaries describe waiting and failures without claiming a finished generation", () => {
  const waiting = canvasToolPresentation("get_canvas_run", { status: "WAITING_SELECTION", totalNodes: 3, nextAction: "internal-tool-name" });
  assert.equal(waiting.tone, "warning");
  assert.match(waiting.summary, /请选择结果/);
  assert.doesNotMatch(JSON.stringify(waiting), /internal-tool-name|nextAction/);
  const failed = canvasToolPresentation("get_canvas_run", { status: "FAILED", totalNodes: 3 });
  assert.equal(failed.tone, "error");
  const invalid = canvasToolPresentation("validate_canvas_run", { status: "blocked", plan: { issues: [{ message: "请选择图片模型" }], cost: {} } });
  assert.equal(invalid.items[0].title, "请选择图片模型");
  assert.equal(invalid.tone, "warning");
});

test("conflict and history reading preserve authored dialogue without exposing JSON structure", () => {
  const text = historyText([{ type: 1, character_name: "阿青", parenthetical: "轻声", content: "这封信来自谁？", sortOrder: 0 }]);
  assert.equal(text, "阿青（轻声）：这封信来自谁？");
  assert.doesNotMatch(text, /character_name|sortOrder|content|[{}]/);
  assert.equal(historyText("请保留用户写下的原话：尚未实现这个愿望。"), "请保留用户写下的原话：尚未实现这个愿望。");
});

test("media import placement preserves the chain and clears the entire batch and nested groups", () => {
  const existing = node("source");
  assert.deepEqual(placements.freeMediaPosition([existing], { x: 380, y: 0 }), { x: 380, y: 0 });
  const image = { ...node("image"), layout: { ...existing.layout, x: 380 } };
  const further = { ...node("further"), layout: { ...existing.layout, x: 760 } };
  assert.deepEqual(placements.freeMediaPosition([further, existing, image], { x: 380, y: 0 }), { x: 1120, y: 0 });
  const lower = { ...node("lower"), layout: { ...existing.layout, x: 600, y: 400 } };
  assert.deepEqual(placements.freeMediaPosition([lower], { x: 380, y: 0 }, 3), { x: 380, y: 0 });
  assert.deepEqual(placements.freeMediaPosition([lower], { x: 380, y: 0 }, 4), { x: 960, y: 0 });
  const group = { ...node("group", "group"), layout: { ...existing.layout, x: 200, y: 100, width: 800, height: 600 } };
  const child = { ...node("child"), layout: { ...existing.layout, parentId: "group", x: 700, y: 40 } };
  assert.deepEqual(placements.freeMediaPosition([child, group], { x: 380, y: 100 }, 2), { x: 1260, y: 100 });
});

test("automatic successors keep the next column and avoid measured tall nodes and occupied lower lanes", () => {
  const source = node("source");
  const preferred = { x: 380, y: 0 };
  const size = { width: 300, height: 384 };
  assert.deepEqual(placements.freeNodePosition([source], preferred, size), preferred);
  const first = { ...node("first"), layout: { ...source.layout, x: 380 } };
  const second = { ...node("second"), layout: { ...source.layout, x: 380, y: 650, height: 260, locked: true } };
  const nextColumn = { ...node("next"), layout: { ...source.layout, x: 760, height: 2000 } };
  const position = placements.freeNodePosition([second, source, nextColumn, first], preferred, size, { first: { width: 300, height: 520 } });
  assert.deepEqual(position, { x: 380, y: 958 });
  assert.equal(first.layout.y, 0);
  assert.equal(second.layout.y, 650);
});

test("writeback successors distinguish the result input from its business target", () => {
  const configured = (id, field) => ({ ...port(id, null), valueTypeField: field });
  const writeback = capability("writeback", [configured("value", "valueType"), configured("target", "targetType")], [], {
    valueType: { required: true, enumValues: ["text", "image", "video", "audio"] },
    targetType: { required: true, enumValues: ["project", "script", "assetItem"] },
  });
  const fromImage = graphTools.compatibleNodes(node("image", "image"), port("media", "image"), [writeback], 1);
  assert.deepEqual(fromImage.map(item => item.port.id), ["value"]);
  assert.equal(fromImage[0].config.valueType, "image");
  const fromProject = graphTools.compatibleNodes(node("project", "project"), port("project", "project"), [writeback], 1);
  assert.deepEqual(fromProject.map(item => item.port.id), ["target"]);
  assert.equal(fromProject[0].config.targetType, "project");
  assert.equal(graphTools.compatibleNodes(node("images", "imageGeneration"), port("images", "image", true), [writeback], 1).length, 0);
});

test("a layout acknowledgement preserves other nodes, media content and edge identities", () => {
  const original = { schemaVersion: 1, projectId: null, nodes: [node("image", "image", { mediaId: "image-media" }), node("text")], edges: [edge("link", "image", "media", "text", "input")] };
  const incoming = structuredClone(original);
  incoming.nodes[0].layout.x = 320;
  incoming.nodes[0].layoutVersion++;
  const result = graphTools.reconcileCanvasGraph(original, incoming);
  assert.notEqual(result.nodes[0], original.nodes[0]);
  assert.equal(result.nodes[0].layout.x, 320);
  assert.equal(result.nodes[0].config, original.nodes[0].config);
  assert.equal(result.nodes[1], original.nodes[1]);
  assert.equal(result.edges, original.edges);
  assert.equal(graphTools.reconcileCanvasGraph(result, structuredClone(result)), result);
  const changed = structuredClone(result);
  changed.nodes[0].config.mediaId = "new-media";
  changed.nodes[0].configVersion++;
  assert.equal(graphTools.reconcileCanvasGraph(result, changed).nodes[0].config.mediaId, "new-media");
  const deleted = graphTools.applyCanvasChanges(result, { upsertedNodes: [], upsertedEdges: [], deletedNodeIds: ["image"], deletedEdgeIds: ["link"] });
  assert.deepEqual(deleted.nodes, [original.nodes[1]]);
  assert.deepEqual(deleted.edges, []);
});

test("asynchronous import commands retain revision order and do not erase an in-flight undo target", async () => {
  const snapshot = { document: { id: "canvas", revision: 3, archived: false }, graph: { schemaVersion: 1, projectId: null, nodes: [], edges: [] }, lastSequence: 3, permissions: { canEdit: true } };
  let finishUndo;
  const loader = moduleLoader({ [path.join(root, "lib/api/canvas.ts")]: { canvasApi: {
    undo: () => new Promise(resolve => { finishUndo = resolve; }), snapshot: async () => structuredClone(snapshot),
  } } });
  const store = loader("lib/store/canvas-store.ts").createCanvasStore("canvas");
  store.setState({ snapshot, catalog: { schemaVersion: 1, nodes: [] }, sync: "connected" });
  const receipt = (id, revision) => ({ commandId: id, canvasId: "canvas", revision, sequence: revision, changes: { upsertedNodes: [], deletedNodeIds: [], upsertedEdges: [], deletedEdgeIds: [] } });
  store.getState().rememberCommand(receipt("late-response", 2));
  store.getState().rememberCommand(receipt("earlier-commit", 1));
  store.getState().rememberCommand(receipt("late-response", 2));
  assert.deepEqual(store.getState().undoStack, ["earlier-commit", "late-response"]);
  const undoing = store.getState().undo();
  store.getState().rememberCommand(receipt("new-import", 4));
  finishUndo(receipt("undo-late-response", 3));
  await undoing;
  assert.deepEqual(store.getState().undoStack, ["earlier-commit", "new-import"]);
  assert.deepEqual(store.getState().redoStack, ["undo-late-response"]);
  store.getState().rememberCommand(receipt("late-response", 2));
  assert.deepEqual(store.getState().redoStack, ["undo-late-response"]);
  assert.throws(() => store.getState().rememberCommand({ ...receipt("wrong", 5), canvasId: "another" }));
  store.getState().reauthorize();
  store.getState().rememberCommand(receipt("late-after-account-change", 6));
  assert.deepEqual(store.getState().undoStack, []);
});

test("save and SSE acknowledgement keep graph content stable while permissions still refresh", async () => {
  const snapshot = { document: { id: "canvas", revision: 1, archived: false }, graph: { schemaVersion: 1, projectId: null, nodes: [node("moving"), node("untouched")], edges: [] }, lastSequence: 1, permissions: { canEdit: true } };
  const moved = { ...structuredClone(snapshot.graph.nodes[0]), layoutVersion: 1, layout: { ...snapshot.graph.nodes[0].layout, x: 80 } };
  const changes = { upsertedNodes: [moved], deletedNodeIds: [], upsertedEdges: [], deletedEdgeIds: [] };
  const receipt = { commandId: "move", canvasId: "canvas", sequence: 2, revision: 2, changes };
  let finishEdit;
  let acknowledgedGraph;
  const api = {
    edit: () => new Promise(resolve => { finishEdit = resolve; }),
    snapshot: async () => ({ ...structuredClone(snapshot), graph: structuredClone(acknowledgedGraph), document: { ...snapshot.document, revision: 2 }, lastSequence: 2, permissions: { canEdit: false } }),
  };
  const loader = moduleLoader({ [path.join(root, "lib/api/canvas.ts")]: { canvasApi: api } });
  const store = loader("lib/store/canvas-store.ts").createCanvasStore("canvas");
  store.setState({ snapshot, catalog: { schemaVersion: 1, nodes: [] } });
  store.getState().select(["moving"]);
  const saving = store.getState().edit("移动节点", [{ type: "canvas.node.layout", nodeId: "moving", expectedLayoutVersion: 0, layout: moved.layout }]);
  assert.equal(store.getState().saving, true);
  assert.equal(store.getState().snapshot.graph, snapshot.graph);
  await store.getState().receive({ canvasId: "canvas", sequence: 2, type: "canvas.graph.changed", payload: receipt, createdAt: "2026-10-13T08:00:00Z" });
  acknowledgedGraph = store.getState().snapshot.graph;
  finishEdit(receipt);
  await saving;
  assert.equal(store.getState().snapshot.graph, acknowledgedGraph);
  assert.equal(store.getState().snapshot.graph.nodes[1], snapshot.graph.nodes[1]);
  assert.equal(store.getState().snapshot.permissions.canEdit, false);
  assert.deepEqual(store.getState().selection, ["moving"]);
  assert.equal(store.getState().saving, false);
});

test("downstream choices retain every compatible port and require an explicit collection selection", () => {
  const catalog = [capability("videoGeneration", [port("firstFrame", "image"), port("lastFrame", "image"), port("references", "image", true)]),
    capability("selection", [{ ...port("candidates", null, true), valueTypeField: "valueType" }], [], { valueType: { required: true, enumValues: ["image", "video"] } })];
  const scalar = graphTools.compatibleNodes(node("source", "image"), port("image", "image"), catalog, null);
  assert.deepEqual(scalar.map(item => item.port.id), ["firstFrame", "lastFrame", "references", "candidates"]);
  const collection = graphTools.compatibleNodes(node("source", "imageGeneration"), port("images", "image", true), catalog, null);
  assert.deepEqual(collection.map(item => item.port.id), ["references", "candidates"]);
  assert.equal(collection[1].config.valueType, "image");
  const projectNode = { ...catalog[0], requiresProject: true };
  assert.equal(graphTools.compatibleNodes(node("source"), port("image", "image"), [projectNode], null).length, 0);
});

test("connection validation rejects literal replacement, duplicate slots, and execution cycles", () => {
  const catalog = [capability("text", [port("prompt", "text")], [port("text", "text")])];
  const graph = { schemaVersion: 1, projectId: null, nodes: [node("a"), node("b")], edges: [] };
  assert.equal(graphTools.validateConnection(graph, catalog, "a", "text", "b", "prompt"), null);
  assert.ok(graphTools.validateConnection({ ...graph, nodes: [node("a"), node("b", "text", { prompt: "保持手填内容" })] }, catalog, "a", "text", "b", "prompt"));
  const connected = { ...graph, edges: [edge("ab", "a", "text", "b", "prompt")] };
  assert.ok(graphTools.validateConnection(connected, catalog, "b", "text", "a", "prompt"));
  assert.ok(graphTools.validateConnection(connected, catalog, "a", "text", "b", "prompt"));
  assert.equal(graphTools.validateConnection({ ...connected, edges: [{ ...connected.edges[0], kind: "reference" }] }, catalog, "b", "text", "a", "prompt"), null);
});

test("copying grouped nodes preserves business references and creates independent operation identities", () => {
  const parent = { ...node("group", "group"), layout: { ...node("group").layout, x: 100, y: 200 } };
  const child = { ...node("child", "asset"), resource: { type: "asset", id: 42, version: 3 }, layout: { ...node("child").layout, x: 20, y: 40, parentId: "group" } };
  const target = node("choice", "selection", { valueType: "image", selectedOutputId: "old-run-choice" });
  const graph = { schemaVersion: 1, projectId: 1, nodes: [parent, child, target], edges: [{ ...edge("old-edge", "child", "image", "choice", "candidates"), output: { nodeRunId: "old-run" } }] };
  const copy = graphTools.copyNodes(graph, ["group", "choice"]);
  const created = copy.filter(item => item.type === "canvas.node.create").map(item => item.node);
  assert.equal(created.length, 3);
  assert.equal(created.find(item => item.kind === "asset").resource.id, 42);
  assert.equal(created.find(item => item.kind === "asset").layout.parentId, created.find(item => item.kind === "group").id);
  assert.equal(created.find(item => item.kind === "selection").config.selectedOutputId, undefined);
  assert.equal(copy.find(item => item.type === "canvas.edge.create").edge.output, null);
  assert.equal(graph.edges[0].output.nodeRunId, "old-run");
  const single = graphTools.copyNodes(graph, ["child"])[0].node;
  assert.equal(single.layout.parentId, null);
  assert.deepEqual({ x: single.layout.x, y: single.layout.y }, { x: 168, y: 288 });
});

test("grouping respects world coordinates and automatic arrangement leaves locked nodes unchanged", () => {
  const a = { ...node("a"), layout: { ...node("a").layout, x: 100, y: 80 } };
  const b = { ...node("b"), layout: { ...node("b").layout, x: 500, y: 300 } };
  const locked = { ...node("locked"), layout: { ...node("locked").layout, locked: true, x: 99 } };
  const graph = { schemaVersion: 1, projectId: null, nodes: [a, b, locked], edges: [] };
  const grouped = layouts.groupCanvasNodes(graph, ["a", "b"]);
  const group = grouped[0].node;
  const aLayout = grouped.find(operation => operation.nodeId === "a").layout;
  assert.equal(group.layout.x + aLayout.x, 100);
  assert.equal(group.layout.y + aLayout.y, 80);
  assert.ok(layouts.arrangeCanvasNodes(graph, [], "arrange").every(operation => operation.nodeId !== "locked"));
  assert.throws(() => layouts.groupCanvasNodes(graph, ["a", "locked"]));
});

test("automatic layout follows links instead of source order and aligns measured ports", () => {
  const a = { ...node("a"), layout: { ...node("a").layout, x: 900, y: 100 } };
  const b = { ...node("b"), layout: { ...node("b").layout, x: 0, y: 400 } };
  const c = { ...node("c"), layout: { ...node("c").layout, x: 400, y: 40 } };
  const graph = { schemaVersion: 1, projectId: null, nodes: [c, b, a], edges: [edge("ab", "a", "out", "b", "in"), edge("bc", "b", "out", "c", "in")] };
  const geometry = {
    a: { width: 300, height: 160, inputs: {}, outputs: { out: 140 } },
    b: { width: 620, height: 400, inputs: { in: 365 }, outputs: { out: 365 } },
    c: { width: 300, height: 210, inputs: { in: 185 }, outputs: {} },
  };
  const positions = new Map(layouts.arrangeCanvasNodes(graph, [], "arrange", geometry).map(operation => [operation.nodeId, operation.layout]));
  assert.ok(positions.get("b").x >= positions.get("a").x + geometry.a.width + 64);
  assert.ok(positions.get("c").x >= positions.get("b").x + geometry.b.width + 64);
  assert.equal(positions.get("a").y + 140, positions.get("b").y + 365);
  assert.equal(positions.get("b").y + 365, positions.get("c").y + 185);
});

test("automatic layout keeps a main chain straight and gives branches separate nonoverlapping lanes", () => {
  const nodes = ["a", "b", "c", "d", "x", "y"].map(id => node(id));
  const graph = { schemaVersion: 1, projectId: null, nodes, edges: [edge("ab", "a", "out", "b", "in"), edge("ac", "a", "out", "c", "in"), edge("bd", "b", "out", "d", "in"), edge("cd", "c", "out", "d", "in"), edge("xy", "x", "out", "y", "in"), edge("ab2", "a", "other", "b", "other")] };
  const positions = new Map(layouts.arrangeCanvasNodes(graph, [], "arrange").map(operation => [operation.nodeId, operation.layout]));
  assert.equal(positions.get("a").y, positions.get("b").y);
  assert.equal(positions.get("b").y, positions.get("d").y);
  assert.ok(positions.get("c").y >= positions.get("b").y + 280 + 64);
  assert.ok(positions.get("d").x > positions.get("c").x);
  assert.equal(positions.get("x").y, positions.get("y").y);
  assert.ok(positions.get("x").y > positions.get("c").y + 280);
});

test("automatic layout preserves group-relative coordinates, locked nodes and unrelated edges", () => {
  const group = { ...node("group", "group"), layout: { ...node("group").layout, x: 1000, y: 700 } };
  const a = { ...node("a"), layout: { ...node("a").layout, x: 40, y: 80, parentId: "group" } };
  const b = { ...node("b"), layout: { ...node("b").layout, x: 40, y: 380, parentId: "group" } };
  const locked = { ...node("fixed"), layout: { ...node("fixed").layout, locked: true } };
  const graph = { schemaVersion: 1, projectId: null, nodes: [group, a, b, locked], edges: [edge("ab", "a", "out", "b", "in"), { ...edge("ba", "b", null, "a", null), kind: "annotation" }] };
  const operations = layouts.arrangeCanvasNodes(graph, ["a", "b", "fixed"], "arrange");
  assert.deepEqual(operations.map(operation => operation.nodeId), ["a", "b"]);
  assert.equal(operations[0].layout.x, 40);
  assert.equal(operations[0].layout.y, 80);
  assert.equal(operations[1].layout.y, 80);
  assert.equal(operations[1].layout.parentId, "group");
  assert.ok(operations[1].layout.x > operations[0].layout.x);
  assert.equal(locked.layout.x, 0);
  assert.equal(graph.edges.length, 2);
});

test("automatic layout reports a cyclic selected topology without emitting unbounded coordinates", () => {
  const graph = { schemaVersion: 1, projectId: null, nodes: [node("a"), node("b")], edges: [edge("ab", "a", "out", "b", "in"), edge("ba", "b", "out", "a", "in")] };
  assert.throws(() => layouts.arrangeCanvasNodes(graph, [], "arrange"), /循环/);
});

test("generated result references and annotation chains stay beside their source", () => {
  const nodes = [node("result"), node("a"), node("b"), node("note")];
  const edges = [edge("ab", "a", "out", "b", "in"), { ...edge("result", "b", null, "result", null), kind: "reference" }, { ...edge("note", "result", null, "note", null), kind: "annotation" }];
  const graph = { schemaVersion: 1, projectId: null, nodes, edges };
  const positions = new Map(layouts.arrangeCanvasNodes(graph, [], "arrange").map(operation => [operation.nodeId, operation.layout]));
  for (const [source, target] of [["a", "b"], ["b", "result"], ["result", "note"]]) {
    assert.ok(positions.get(target).x > positions.get(source).x);
    assert.equal(positions.get(target).y, positions.get(source).y);
  }
  assert.deepEqual(graph.edges, edges);
});

test("cyclic annotations cannot reverse execution direction or prevent arranging", () => {
  const graph = { schemaVersion: 1, projectId: null, nodes: [node("a"), node("b"), node("c")], edges: [
    { ...edge("back", "c", null, "a", null), kind: "annotation" },
    edge("ab", "a", "out", "b", "in"), edge("bc", "b", "out", "c", "in"),
  ] };
  const positions = new Map(layouts.arrangeCanvasNodes(graph, [], "arrange").map(operation => [operation.nodeId, operation.layout]));
  assert.ok(positions.get("a").x < positions.get("b").x && positions.get("b").x < positions.get("c").x);
  assert.equal(graph.edges.length, 3);
});

test("selection toolbar follows the selection and stays inside the canvas near every edge", () => {
  const { selectionToolbarPosition } = load("lib/canvas/selection-toolbar.ts");
  const size = { width: 300, height: 44 };
  const initial = { left: 400, right: 700, top: 220, bottom: 420, width: 1200, height: 800 };
  const first = selectionToolbarPosition(initial, size);
  const moved = selectionToolbarPosition({ ...initial, left: 480, right: 780, top: 300, bottom: 500 }, size);
  assert.equal(moved.left - first.left, 80);
  assert.equal(moved.top - first.top, 80);
  assert.ok(first.top + size.height < initial.top);
  const nearTop = selectionToolbarPosition({ ...initial, top: 4, bottom: 180 }, size);
  assert.ok(nearTop.top > 180);
  for (const bounds of [{ ...initial, left: -180, right: 0 }, { ...initial, left: 1190, right: 1490 }, { ...initial, top: -500, bottom: 1200 }]) {
    const position = selectionToolbarPosition(bounds, size);
    assert.ok(position.left >= 8 && position.left + size.width <= bounds.width - 8);
    assert.ok(position.top >= 8 && position.top + size.height <= bounds.height - 8);
  }
});

test("unconnected media packs into rows instead of creating a tall column", () => {
  const graph = { schemaVersion: 1, projectId: null, nodes: ["a", "b", "c", "d"].map(id => node(id)), edges: [] };
  const positions = new Map(layouts.arrangeCanvasNodes(graph, [], "arrange").map(operation => [operation.nodeId, operation.layout]));
  assert.equal(positions.get("a").y, positions.get("b").y);
  assert.equal(positions.get("c").y, positions.get("d").y);
  assert.ok(positions.get("c").y > positions.get("a").y + 280);
});

test("business save acknowledgements preserve typing and reverts made while a request was pending", () => {
  const baseline = { header: { resourceType: "project", resourceId: 1, projectId: 1, version: 0, deleted: false }, data: { name: "原名", description: "原描述" } };
  const submitted = { baseline, patch: { name: "提交的名称", description: "提交的描述" }, requestId: "request" };
  const current = { baseline, patch: { description: "保存中继续输入" }, requestId: "new-draft" };
  const acknowledged = { ...baseline, header: { ...baseline.header, version: 1 }, data: { ...baseline.data, ...submitted.patch } };
  const result = drafts.acknowledgeResourceDraft(current, submitted, acknowledged);
  assert.deepEqual(result.patch, { name: "原名", description: "保存中继续输入" });
  assert.equal(result.baseline.header.version, 1);
  assert.deepEqual(drafts.conflictingFields(submitted, { ...acknowledged, data: { name: "别人改名", description: "原描述" } }), ["name"]);
});

test("structured fields are validated before submission without silently erasing invalid drafts", () => {
  assert.throws(() => drafts.normalizeResourcePatch("project", { properties: "{broken" }));
  assert.throws(() => drafts.normalizeResourcePatch("project", { properties: "[]" }));
  assert.deepEqual(drafts.normalizeResourcePatch("asset", { tags: ["雨夜", "", "  角色  "] }), { tags: ["雨夜", "角色"] });
});

test("canvas SSE rejects mismatched event IDs and exposes revocation as an explicit terminal error", () => {
  const event = { canvasId: "canvas", sequence: 4, type: "canvas.graph.changed", payload: { revision: 4 } };
  assert.equal(streams.parseCanvasEvent(": heartbeat"), null);
  assert.equal(streams.parseCanvasEvent(`id: 4\nevent: canvas.graph.changed\ndata: ${JSON.stringify(event)}`).sequence, 4);
  assert.throws(() => streams.parseCanvasEvent(`id: 3\nevent: canvas.graph.changed\ndata: ${JSON.stringify(event)}`));
  assert.throws(() => streams.parseCanvasEvent('event: canvas.error\ndata: {"errorCode":"ACCESS_DENIED","message":"撤权"}'), error => error.code === "ACCESS_DENIED");
});

test("ambiguous mutation retry uses the original request identity and records exactly one undo entry", async () => {
  const snapshot = { document: { id: "canvas", revision: 1, archived: false }, graph: { schemaVersion: 1, projectId: null, nodes: [], edges: [] }, lastSequence: 1, permissions: { canEdit: true } };
  let snapshotReads = 0;
  const edits = [];
  const mockedApi = {
    capabilities: async () => ({ schemaVersion: 1, nodes: [] }),
    snapshot: async () => { if (++snapshotReads === 1) throw new Error("response lost after commit"); return snapshot; },
    edit: async (_id, command) => { edits.push(command); return { commandId: "once", changes: {}, revision: 1 }; },
  };
  const loader = moduleLoader({ [path.join(root, "lib/api/canvas.ts")]: { canvasApi: mockedApi } });
  const { createCanvasStore } = loader("lib/store/canvas-store.ts");
  const store = createCanvasStore("canvas");
  store.setState({ snapshot, catalog: { schemaVersion: 1, nodes: [] } });
  await assert.rejects(store.getState().edit("创建文本", [{ type: "canvas.node.create", node: node("text") }]));
  assert.ok(store.getState().pending);
  await store.getState().retry();
  assert.equal(edits[0].requestId, edits[1].requestId);
  assert.deepEqual(store.getState().undoStack, ["once"]);
  assert.equal(store.getState().pending, null);
});

test("definitive command rejection releases the canvas for the next edit", async () => {
  const snapshot={document:{id:"canvas",revision:1,archived:false},graph:{schemaVersion:1,projectId:null,nodes:[],edges:[]},lastSequence:1,permissions:{canEdit:true}};
  let attempts=0;
  const api={snapshot:async()=>snapshot,edit:async()=>{
    if(++attempts===1) throw {isAxiosError:true,response:{status:409,data:{msg:"内容已变化，请重新修改"}}};
    return {commandId:"next",revision:2,changes:{}};
  }};
  const loader=moduleLoader({[path.join(root,"lib/api/canvas.ts")]:{canvasApi:api}});
  const store=loader("lib/store/canvas-store.ts").createCanvasStore("canvas");
  store.setState({snapshot,catalog:{schemaVersion:1,nodes:[]}});
  await assert.rejects(store.getState().edit("第一次",[{type:"canvas.node.create",node:node("a")} ]));
  assert.equal(store.getState().pending,null);
  assert.equal(store.getState().saving,false);
  await store.getState().edit("下一次",[{type:"canvas.node.create",node:node("b")}]);
  assert.deepEqual(store.getState().undoStack,["next"]);
});

test("business rejection codes survive HTTP success envelopes while acknowledged commands stay recoverable", async () => {
  assert.equal(apiErrors.getApiErrorStatus(new apiErrors.ApiResultError(400,"配置无效")),400);
  const snapshot={document:{id:"canvas",revision:1,archived:false},graph:{schemaVersion:1,projectId:null,nodes:[],edges:[]},lastSequence:1,permissions:{canEdit:true}};
  const api={edit:async()=>({commandId:"accepted",revision:2}),snapshot:async()=>{throw {isAxiosError:true,response:{status:400,data:{msg:"快照读取失败"}}};}};
  const loader=moduleLoader({[path.join(root,"lib/api/canvas.ts")]:{canvasApi:api}});
  const store=loader("lib/store/canvas-store.ts").createCanvasStore("canvas");
  store.setState({snapshot,catalog:{schemaVersion:1,nodes:[]}});
  await assert.rejects(store.getState().edit("已提交",[{type:"canvas.node.create",node:node("a")} ]));
  assert.ok(store.getState().pending);
  assert.deepEqual(store.getState().undoStack,[]);
});

test("clipboard exports only the selected subgraph and rejects malformed fragments", () => {
  const graph = {schemaVersion:1,projectId:1,nodes:[node("selected","text",{content:"选区"}),node("private","text",{content:"不应被复制的其他内容"})],edges:[]};
  const encoded = clipboard.encodeCanvasClipboard(graph,["selected"]);
  assert.ok(!encoded.includes("不应被复制的其他内容"));
  const fragment = clipboard.decodeCanvasClipboard(encoded);
  assert.equal(fragment.nodes.length,1);
  assert.notEqual(fragment.nodes[0].id,"selected");
  assert.throws(() => clipboard.decodeCanvasClipboard('{"format":"afv-canvas-1","graph":{"schemaVersion":1,"projectId":null,"nodes":[{}],"edges":[]}}'));
  assert.equal(placements.mediaNode({id:"pdf",name:"参考稿.pdf",state:"READY",content:{kind:"DOCUMENT"}},{x:20,y:30}).kind,"document");
  assert.equal(placements.mediaNode({id:"text",name:"原文.txt",state:"READY",content:{kind:"TEXT"}},{x:20,y:30}).kind,"textFile");
});

test("a scope change discards in-flight snapshots from the previous authorization", async () => {
  const snapshot = {document:{id:"canvas",revision:1,archived:false},graph:{schemaVersion:1,projectId:null,nodes:[],edges:[]},lastSequence:1,permissions:{canEdit:true}};
  let resolveRead;
  const api = {capabilities:async()=>({schemaVersion:1,nodes:[]}),snapshot:()=>new Promise(resolve => {resolveRead=resolve;})};
  const loader=moduleLoader({[path.join(root,"lib/api/canvas.ts")]:{canvasApi:api}});
  const store=loader("lib/store/canvas-store.ts").createCanvasStore("canvas");
  const read=store.getState().load();
  store.getState().reauthorize();
  resolveRead(snapshot);
  await read;
  assert.equal(store.getState().snapshot,null);
  assert.equal(store.getState().catalog,null);
});

test("a mutation acknowledgement after scope change cannot repopulate document or undo state", async () => {
  const snapshot = {document:{id:"canvas",revision:1,archived:false},graph:{schemaVersion:1,projectId:null,nodes:[],edges:[]},lastSequence:1,permissions:{canEdit:true}};
  let resolveEdit;
  const api={edit:()=>new Promise(resolve=>{resolveEdit=resolve;}),snapshot:()=>{throw new Error("old scope must not reload");}};
  const loader=moduleLoader({[path.join(root,"lib/api/canvas.ts")]:{canvasApi:api}});
  const store=loader("lib/store/canvas-store.ts").createCanvasStore("canvas");
  store.setState({snapshot,catalog:{schemaVersion:1,nodes:[]}});
  const mutation=store.getState().edit("创建",[{type:"canvas.node.create",node:node("a")}]);
  const rejected=assert.rejects(mutation,/权限范围已经变化/);
  store.getState().reauthorize();
  resolveEdit({commandId:"old-scope",changes:{},revision:2});
  await rejected;
  assert.equal(store.getState().snapshot,null);
  assert.equal(store.getState().pending,null);
  assert.deepEqual(store.getState().undoStack,[]);
});


test("shot frames remain discoverable and connectable as images before configuring a model", () => {
  const constraints=load("lib/canvas/model-constraints.ts");
  const {canvasConnectionIssue}=load("lib/canvas/connection-readiness.ts");
  const source=node("shot","shot");
  const target=node("generator","imageGeneration");
  const shotType=capability("shot",[],[port("firstFrame","image"),port("lastFrame","image"),port("video","video")]);
  const imageType=capability("imageGeneration",[port("referenceImages","image",true),port("prompt","text")]);
  const catalog={schemaVersion:1,nodes:[shotType,imageType],models:[],audioServices:[]};
  const graph={schemaVersion:1,projectId:1,nodes:[source,target],edges:[]};
  for(const frame of ["firstFrame","lastFrame"]) {
    assert.ok(graphTools.compatibleNodes(source,shotType.outputs.find(item=>item.id===frame),catalog.nodes,1)
      .some(item=>item.capability.kind==="imageGeneration"&&item.port.id==="referenceImages"));
    assert.equal(graphTools.validateConnection(graph,catalog.nodes,source.id,frame,target.id,"referenceImages"),null);
  }
  assert.match(graphTools.validateConnection(graph,catalog.nodes,source.id,"video",target.id,"referenceImages"),/类型不兼容/);
  graph.edges=[edge("frame",source.id,"firstFrame",target.id,"referenceImages")];
  assert.match(constraints.validateModelNode(target,graph,catalog),/选择.*模型/);
  target.config={modelId:9};
  catalog.models=[{id:9,name:"文本生图",kind:"image",resultType:"image",inputs:{prompt:{maximum:1,minimum:0}},parameters:{},versions:[],unavailable:null}];
  assert.match(constraints.validateModelNode(target,graph,catalog),/不支持参考图片/);
  assert.match(canvasConnectionIssue(graph.edges[0],graph,catalog),/不支持参考图片/);
  catalog.models[0].inputs.referenceImages={maximum:2,minimum:0};
  assert.equal(constraints.validateModelNode(target,graph,catalog),null);
  assert.equal(canvasConnectionIssue(graph.edges[0],graph,catalog),null);
  graph.edges.push(edge("last",source.id,"lastFrame",target.id,"referenceImages"));
  catalog.models[0].inputs.referenceImages.maximum=1;
  assert.match(canvasConnectionIssue(graph.edges[0],graph,catalog),/最多允许 1/);
});

test("unconfigured nodes have readable readiness states while partial fields remain saveable", () => {
  const {canvasNodeConfigurationIssue}=load("lib/canvas/node-readiness.ts");
  const {validateCanvasConfig}=load("lib/canvas/config.ts");
  const type=capability("collection",[],[],{valueType:{type:"string",required:true,enumValues:["text","image"],minimum:null,maximum:null,maxLength:20}});
  assert.equal(validateCanvasConfig(type,{},false),null);
  assert.match(validateCanvasConfig(type,{}),/内容类型/);
  assert.match(validateCanvasConfig(type,{valueType:"invalid"},false),/允许选项/);
  const image=node("file","image");
  const graph={schemaVersion:1,projectId:null,nodes:[image],edges:[]};
  const catalog={schemaVersion:1,nodes:[],models:[],audioServices:[]};
  assert.equal(canvasNodeConfigurationIssue(image,{...capability("image",[]),label:"图片"},catalog,graph),"请选择图片文件");
  const source=node("text","text");
  assert.equal(canvasNodeConfigurationIssue(source,{...capability("text",[]),label:"文本"},catalog,graph),"请填写文本内容");
});

test("model readiness rejects count, mode and version mismatches without limiting node discovery", () => {
  const models = load("lib/canvas/model-constraints.ts");
  const source = node("source", "imageGeneration", {modelId:1,count:4});
  const target = node("target", "videoGeneration", {modelId:2});
  const catalog = {schemaVersion:1,nodes:[capability("imageGeneration",[],[port("images","image",true)]),capability("videoGeneration",[port("referenceImages","image",true),port("firstFrame","image")])],models:[
    {id:2,name:"reference model",kind:"video",resultType:"video",inputs:{referenceImages:{maximum:3,minimum:0},firstFrame:{maximum:1,minimum:0}},parameters:{exclusiveInputModes:true},versions:[],unavailable:null}
  ]};
  const graph={schemaVersion:1,projectId:null,nodes:[source,target],edges:[edge("ref","source","images","target","referenceImages")]};
  assert.match(models.validateModelNode(target,graph,catalog),/最多允许 3/);
  source.config.count=2;
  assert.equal(models.validateModelNode(target,graph,catalog),null);
  graph.edges.push(edge("first","source","images","target","firstFrame"));
  source.config.count=1;
  assert.match(models.validateModelNode(target,graph,catalog),/不能同时/);
  catalog.models[0].unavailable="供应商停用";
  assert.match(models.validateModelNode(target,graph,catalog),/停用/);
  assert.equal(models.modelsForNode("videoGeneration",catalog).length,0);
});

test("ComfyUI version limits and conflicts are shared by node validation and connection feedback", () => {
  const models = load("lib/canvas/model-constraints.ts");
  const { canvasConnectionIssue } = load("lib/canvas/connection-readiness.ts");
  const source = node("source", "imageGeneration", { count: 4 });
  const target = node("target", "comfyui", { modelId: 2, workflowVersionId: 21, resultType: "video" });
  const inputs = { referenceImages: { maximum: 4, minimum: 0, minimumWhenProvided: 4 } };
  const catalog = { nodes: [capability("imageGeneration", [], [port("images", "image", true)]),
    capability("comfyui", [port("referenceImages", "image", true)])], models: [{ id: 2, kind: "comfyui",
    resultType: "video", inputs: {}, parameters: {}, unavailable: null, versions: [
      { id: 20, number: 1, inputs: {}, fields: [], parameters: {}, unavailable: "此版本与使用上限冲突" },
      { id: 21, number: 2, inputs, fields: ["referenceImages"], parameters: { maxImageInputs: 4 }, unavailable: null },
    ] }] };
  const connection = edge("ref", "source", "images", "target", "referenceImages");
  const graph = { nodes: [source, target], edges: [connection] };
  assert.equal(models.validateModelNode(target, graph, catalog), null);
  source.config.count = 5;
  assert.match(models.validateModelNode(target, graph, catalog), /最多允许 4/);
  assert.match(canvasConnectionIssue(connection, graph, catalog), /最多允许 4/);
  target.config.workflowVersionId = 20;
  assert.match(models.validateModelNode(target, graph, catalog), /此版本.*冲突/);
  assert.match(canvasConnectionIssue(connection, graph, catalog), /此版本.*冲突/);
  assert.equal(models.modelsForNode("comfyui", catalog).length, 1);
});

test("node drafts merge independent fields and make same-field changes explicit", () => {
  const tools=load("lib/canvas/node-draft.ts");
  const base=node("n","text",{content:"before",unused:"clear me"});
  const draft={base,title:"my title",config:{content:"mine"}};
  const current={...base,title:"server title",config:{content:"before",unused:"clear me",newField:"remote"},configVersion:2};
  assert.deepEqual(tools.nodeDraftConflicts(draft,current),["@title"]);
  const merged=tools.rebaseNodeDraft(draft,current,["@title"]);
  assert.equal(merged.title,"server title");
  assert.deepEqual(merged.config,{content:"mine",newField:"remote"});
  assert.equal(merged.base.configVersion,2);
  assert.deepEqual(base.config,{content:"before",unused:"clear me"});
});

test("business relationship display never adds execution edges or invents missing objects", () => {
  const {canvasBusinessRelations}=load("lib/canvas/relations.ts");
  const parent={...node("parent"),resource:{type:"script",id:1,version:0}};
  const child={...node("child"),resource:{type:"scriptEpisode",id:2,version:0}};
  const asset={...node("asset"),resource:{type:"asset",id:3,version:0}};
  const graph={schemaVersion:1,projectId:1,nodes:[parent,child,asset],edges:[]};
  const library=[{ref:child.resource,parent:"script:1",relations:[{key:"asset:3",label:"角色"},{key:"asset:999",label:"角色"}]}];
  assert.equal(canvasBusinessRelations(graph,library,"all").length,2);
  assert.equal(canvasBusinessRelations(graph,library,"structure")[0].source,"parent");
  assert.equal(canvasBusinessRelations(graph,library,"assets")[0].source,"asset");
  assert.deepEqual(graph.edges,[]);
});

test("text uploads declare a supported format while bytes and file identity stay intact", async () => {
  const {canvasUploadFile}=load("lib/canvas/files.ts");
  const raw=new File(["# 测试\n正文"],"script.md",{type:"",lastModified:42});
  const prepared=canvasUploadFile(raw);
  assert.equal(prepared.type,"text/markdown");
  assert.equal(await prepared.text(),await raw.text());
  assert.equal(prepared.lastModified,42);
  const video=new File(["fake"],"clip.mp4",{type:"video/mp4"});
  assert.equal(canvasUploadFile(video),video);
});


test("clearing resource authorization rejects late mutation acknowledgements", () => {
  const {createCanvasResourceCache}=load("lib/canvas/resource-cache.ts");
  const cache=createCanvasResourceCache("c");
  const resource={header:{resourceType:"script",resourceId:1,version:2,projectId:1},data:{sourceText:"private"},canEdit:true};
  const ref={type:"script",id:1,version:2};
  const before=cache.generation();
  cache.accept(resource,before);
  assert.equal(cache.read(ref).data.data.sourceText,"private");
  cache.clear();
  cache.accept(resource,before);
  assert.equal(cache.read(ref).data,null);
});


test("audio nodes can be placed before choosing a service but remain unconfigured", () => {
 const model=load("lib/canvas/model-constraints.ts");
 const catalog={schemaVersion:1,nodes:[],models:[],audioServices:[]};
 const audio=node("music","musicGeneration",{apiConfigId:12});
 const graph={schemaVersion:1,projectId:null,nodes:[audio],edges:[]};
 assert.match(model.validateModelNode(audio,graph,catalog),/音频服务/);
 catalog.audioServices=[{id:12,name:"Music"}];
 assert.equal(model.validateModelNode(audio,graph,catalog),null);
});


test("model pickers account for known collection counts beyond model limits", () => {
 const constraints=load("lib/canvas/model-constraints.ts");
 const model={id:2,name:"three references",kind:"video",resultType:"video",inputs:{referenceImages:{minimum:0,maximum:3}},parameters:{},versions:[],unavailable:null};
 const catalog={schemaVersion:1,nodes:[capability("imageGeneration",[],[port("images","image",true)])],models:[model],audioServices:[]};
 const source=node("images","imageGeneration",{count:4});
 const graph={schemaVersion:1,projectId:null,nodes:[source],edges:[]};
 const count=constraints.knownCount(graph,catalog,source.id,"images");
 assert.equal(count,4);
 const larger={...model,id:3,inputs:{referenceImages:{minimum:0,maximum:8}}};
 catalog.models.push(larger);
 assert.equal(constraints.modelFitsInputs(model,model.inputs,["referenceImages"],{referenceImages:count}),false);
 assert.equal(constraints.modelFitsInputs(larger,larger.inputs,["referenceImages"],{referenceImages:count}),true);
 assert.equal(constraints.modelFitsInputs({...larger,parameters:{maxImageInputs:3}},larger.inputs,["referenceImages"],{referenceImages:4}),false);
});
