import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import ts from "typescript";

const source = fs.readFileSync(path.resolve(import.meta.dirname, "../lib/canvas/generation.ts"), "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText;
const loaded = { exports: {} };
vm.runInNewContext(compiled, { module: loaded, exports: loaded.exports, Map, Set });
const {
  missingGenerationSources,
  textGenerationWriteback,
  canvasNodePanelKind,
  hasCanvasQuickPanel,
} = loaded.exports;
const catalog = { nodes: [
  { kind: "text", executor: "constant" },
  { kind: "image", executor: "resource" },
  { kind: "imageGeneration", executor: "image" },
  { kind: "videoGeneration", executor: "video" },
  { kind: "textCombine", executor: "text" },
] };
const node = (id, kind) => ({ id, kind, title: id });
const edge = (sourceNodeId, targetNodeId, output = null) => ({ kind: "execution", sourceNodeId, targetNodeId, output });
const ready = (nodeId, changes = {}) => ({ node: {
  nodeId, status: "SUCCEEDED", derivedState: "CURRENT", contentUnavailable: false, ...changes,
} });

// 防止一次生视频操作隐式重做上游生图，产生额外任务和费用。
test("生视频前提示处理尚未完成的上游生图节点", () => {
  const graph = { nodes: [node("picture", "imageGeneration"), node("video", "videoGeneration")],
    edges: [edge("picture", "video")] };
  assert.deepEqual(Array.from(missingGenerationSources(graph, catalog, "video", []), (item) => item.id), ["picture"]);
  assert.equal(missingGenerationSources(graph, catalog, "video", [ready("picture")]).length, 0);
  assert.equal(missingGenerationSources(graph, catalog, "video", [ready("picture", { derivedState: "STALE" })]).length, 1);
  assert.equal(missingGenerationSources(graph, catalog, "video", [ready("picture", { contentUnavailable: true })]).length, 1);
});

test("已有文本、图片与文本拼接无需先提交独立任务", () => {
  const graph = { nodes: [node("text", "text"), node("combine", "textCombine"), node("image", "image"), node("target", "imageGeneration")],
    edges: [edge("text", "combine"), edge("combine", "target"), edge("image", "target")] };
  assert.equal(missingGenerationSources(graph, catalog, "target", []).length, 0);
});

test("使用固定结果时不要求重新生成来源节点", () => {
  const graph = { nodes: [node("picture", "imageGeneration"), node("video", "videoGeneration")],
    edges: [edge("picture", "video", { outputId: "chosen" })] };
  assert.equal(missingGenerationSources(graph, catalog, "video", []).length, 0);
});

test("可复用结果截断其更早的执行依赖", () => {
  const graph = { nodes: [node("older", "imageGeneration"), node("picture", "imageGeneration"), node("video", "videoGeneration")],
    edges: [edge("older", "picture"), edge("picture", "video")] };
  assert.equal(missingGenerationSources(graph, catalog, "video", [ready("picture")]).length, 0);
});

const written = (text, changes = {}) => ({ node: {
  status: "SUCCEEDED", derivedState: "CURRENT", contentUnavailable: false,
  outputs: { values: { text: [{ text }] } }, ...changes,
} });
const promptEdge = (sourceNodeId, targetNodeId) => ({ kind: "execution", sourceNodeId, targetNodeId, targetPort: "prompt" });

// AI 写作结果只写回提供写作素材的文本节点，防止覆盖其他节点或无关正文。
test("生成文本结果写回连到创作要求的文本节点", () => {
  const graph = { nodes: [node("draft", "text"), node("writer", "textGeneration")], edges: [promptEdge("draft", "writer")] };
  const result = textGenerationWriteback(graph, "writer", written("扩写后的段落"));
  assert.equal(result.target.id, "draft");
  assert.equal(result.text, "扩写后的段落");
  const combined = { nodes: [node("combine", "textCombine"), node("writer", "textGeneration")], edges: [promptEdge("combine", "writer")] };
  assert.equal(textGenerationWriteback(combined, "writer", written("段落")), null);
  assert.equal(textGenerationWriteback({ nodes: [node("writer", "textGeneration")], edges: [] }, "writer", written("段落")), null);
});

// 写回后输入变化、结果转为待更新；过期或不可读的结果不能再覆盖正文。
test("过期、未完成或为空的生成结果不提供写回", () => {
  const graph = { nodes: [node("draft", "text"), node("writer", "textGeneration")], edges: [promptEdge("draft", "writer")] };
  assert.equal(textGenerationWriteback(graph, "writer", written("段落", { derivedState: "STALE" })), null);
  assert.equal(textGenerationWriteback(graph, "writer", written("段落", { status: "RUNNING" })), null);
  assert.equal(textGenerationWriteback(graph, "writer", written("段落", { contentUnavailable: true })), null);
  assert.equal(textGenerationWriteback(graph, "writer", written("  ")), null);
  assert.equal(textGenerationWriteback(graph, "writer", undefined), null);
});

// 所有节点均通过节点下方悬浮面板编辑，按节点能力分派对应悬浮表单，不再弹出右侧属性栏。
test("所有画布节点分派到对应悬浮面板形态", () => {
  for (const kind of ["imageGeneration", "videoGeneration", "textGeneration"]) {
    const item = node(kind, kind);
    assert.equal(hasCanvasQuickPanel(item), true);
    assert.equal(canvasNodePanelKind(item), "generation");
  }
  for (const kind of ["text", "story", "note"]) {
    const item = node(kind, kind);
    assert.equal(hasCanvasQuickPanel(item), true);
    assert.equal(canvasNodePanelKind(item), "text");
  }
  for (const kind of ["image", "video", "audio", "textFile", "document"]) {
    const item = node(kind, kind);
    assert.equal(hasCanvasQuickPanel(item), true);
    assert.equal(canvasNodePanelKind(item), "media");
  }
  for (const kind of ["imageTransform", "agent", "mcpTool", "group", "script", "shot"]) {
    const item = node(kind, kind);
    assert.equal(hasCanvasQuickPanel(item), true);
    assert.equal(canvasNodePanelKind(item), "config");
  }
  const storyGraph = {
    nodes: [node("story", "story"), node("writer", "textGeneration")],
    edges: [promptEdge("story", "writer")],
  };
  assert.equal(textGenerationWriteback(storyGraph, "writer", written("扩写后的故事"))?.target.id, "story");
});
